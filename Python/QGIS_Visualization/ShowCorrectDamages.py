"""
PyQGISスクリプト: 推論結果CSVから正解画像に対応するFGBファイルを結合

このスクリプトは、画像分類モデルの推論結果（CSV）を参照し、
正解（Correct=TRUE）と判定された画像に対応するポリゴンデータ（.fgb）を
特定し、それらを1つのレイヤに統合して書き出します。
"""

import os
from pathlib import Path
import pandas as pd
from qgis.core import (
    QgsApplication,
    QgsVectorLayer,
    QgsProject,
    QgsVectorFileWriter,
    QgsCoordinateReferenceSystem
)
from qgis import processing
from qgis.analysis import QgsNativeAlgorithms

# QGISアプリケーションの初期化（スタンドアロン実行時）
# QGISコンソールから実行する場合は、この部分をコメントアウトしてください
# QgsApplication.setPrefixPath("C:/Program Files/QGIS 3.x/apps/qgis", True)
# qgs = QgsApplication([], False)
# qgs.initQgis()
# processing.initialize()
# QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())


def load_and_filter_csv(csv_path, correct_column='Correct', filename_column='filename'):
    """
    CSVファイルを読み込み、Correct==TRUEの行を抽出してfilenameリストを返す
    
    Parameters:
    -----------
    csv_path : str
        CSVファイルのパス
    correct_column : str
        Correct列の名前（デフォルト: 'Correct'）
    filename_column : str
        filename列の名前（デフォルト: 'filename'）
    
    Returns:
    --------
    list
        .fgbファイル名のリスト（拡張子付き）
    """
    # CSVの読み込み
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        # utf-8-sigで失敗した場合はutf-8で再試行
        df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 必要な列の存在確認
    if correct_column not in df.columns:
        raise ValueError(f"CSVに'{correct_column}'列が見つかりません。列名: {list(df.columns)}")
    if filename_column not in df.columns:
        raise ValueError(f"CSVに'{filename_column}'列が見つかりません。列名: {list(df.columns)}")
    
    # Correct列の値を正規化（文字列'TRUE'/'True'/'true'やブール値Trueを統一）
    def normalize_boolean(value):
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.upper() in ['TRUE', '1', 'YES', 'Y']
        return bool(value)
    
    df['_correct_normalized'] = df[correct_column].apply(normalize_boolean)
    
    # Correct==TRUEの行を抽出
    correct_df = df[df['_correct_normalized'] == True]
    
    if len(correct_df) == 0:
        print("警告: Correct==TRUEの行が見つかりませんでした。")
        return []
    
    # filenameから拡張子を.pngから.fgbに変更
    fgb_filenames = []
    for filename in correct_df[filename_column]:
        if pd.isna(filename):
            continue
        filename_str = str(filename)
        # .png拡張子を.fgbに置換
        if filename_str.lower().endswith('.png'):
            fgb_filename = filename_str[:-4] + '.fgb'
        else:
            # .png拡張子がない場合は.fgbを追加
            fgb_filename = filename_str + '.fgb'
        fgb_filenames.append(fgb_filename)
    
    print(f"正解画像に対応するFGBファイル数: {len(fgb_filenames)}")
    return fgb_filenames


def find_fgb_files(fgb_filenames, search_directory):
    """
    指定ディレクトリ内から対応する.fgbファイルのパスを取得
    
    Parameters:
    -----------
    fgb_filenames : list
        .fgbファイル名のリスト
    search_directory : str
        検索対象ディレクトリのパス
    
    Returns:
    --------
    list
        見つかった.fgbファイルのフルパスのリスト
    """
    search_path = Path(search_directory)
    if not search_path.exists():
        raise FileNotFoundError(f"検索ディレクトリが見つかりません: {search_directory}")
    
    found_files = []
    for fgb_filename in fgb_filenames:
        fgb_path = search_path / fgb_filename
        if fgb_path.exists():
            found_files.append(str(fgb_path))
        else:
            print(f"警告: ファイルが見つかりません: {fgb_path}")
    
    print(f"見つかったFGBファイル数: {len(found_files)} / {len(fgb_filenames)}")
    return found_files


def merge_vector_layers(fgb_file_paths, output_path):
    """
    複数のFGBファイルを読み込んで1つのレイヤに統合し、出力
    
    Parameters:
    -----------
    fgb_file_paths : list
        .fgbファイルのパスのリスト
    output_path : str
        出力ファイルのパス（.fgb形式）
    
    Returns:
    --------
    bool
        成功した場合True
    """
    if not fgb_file_paths:
        print("エラー: マージするファイルがありません。")
        return False
    
    # 出力ディレクトリが存在しない場合は作成
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各FGBファイルをレイヤとして読み込む
    layers = []
    layer_ids_to_remove = []  # 後で削除するためのIDリスト
    
    for fgb_path in fgb_file_paths:
        layer = QgsVectorLayer(fgb_path, Path(fgb_path).stem, "ogr")
        if not layer.isValid():
            print(f"警告: レイヤの読み込みに失敗しました: {fgb_path}")
            continue
        
        # レイヤを一時的にプロジェクトに追加（native:mergevectorlayersがIDを必要とするため）
        QgsProject.instance().addMapLayer(layer, False)
        layers.append(layer)
        layer_ids_to_remove.append(layer.id())
    
    if not layers:
        print("エラー: 有効なレイヤが1つもありません。")
        return False
    
    print(f"読み込んだレイヤ数: {len(layers)}")
    
    # 最初のレイヤのCRSを取得（すべてのレイヤが同じCRSであることを想定）
    crs = layers[0].crs()
    
    # マージ処理の実行
    try:
        # レイヤIDのリストを作成（プロジェクトに追加済み）
        layer_ids = [layer.id() for layer in layers]
        
        # native:mergevectorlayersアルゴリズムを使用
        result = processing.run(
            "native:mergevectorlayers",
            {
                'LAYERS': layer_ids,
                'CRS': crs,
                'OUTPUT': output_path
            }
        )
        
        # 一時的に追加したレイヤをプロジェクトから削除
        for layer_id in layer_ids_to_remove:
            layer_to_remove = QgsProject.instance().mapLayer(layer_id)
            if layer_to_remove:
                QgsProject.instance().removeMapLayer(layer_id)
        
        if result and 'OUTPUT' in result:
            output_file = result['OUTPUT']
            print(f"マージ完了: {output_file}")
            
            # 出力ファイルの情報を表示
            merged_layer = QgsVectorLayer(output_file, "merged_layer", "ogr")
            if merged_layer.isValid():
                feature_count = merged_layer.featureCount()
                print(f"統合された地物数: {feature_count}")
                merged_layer = None  # 参照を解放
                return True
            else:
                print("警告: 出力レイヤが無効です。")
                return False
        else:
            print("エラー: マージ処理が失敗しました。")
            return False
            
    except Exception as e:
        # エラーが発生した場合も、追加したレイヤを削除
        for layer_id in layer_ids_to_remove:
            layer_to_remove = QgsProject.instance().mapLayer(layer_id)
            if layer_to_remove:
                QgsProject.instance().removeMapLayer(layer_id)
        
        print(f"エラー: マージ処理中に例外が発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    メイン処理関数
    
    設定パラメータをここで変更してください。
    """
    # ========== 設定パラメータ ==========
    # 推論結果CSVファイルのパス
    csv_path = r"C:\Users\kyohe\Aerial_Photo_Classifier\20251209Data\Results\20251212_1758\20251212_1758_forward_ep150.csv"
    
    # FGBファイルが格納されているディレクトリ
    fgb_directory = r"C:\Users\kyohe\Aerial_Photo_Classifier\20251209Data\SquarePolygons\house_collapse\wajima_all"
    
    # 出力ファイルのパス
    output_path = r"C:\Users\kyohe\Aerial_Photo_Classifier\20251209Data\Results\20251212_1758\merged_correct_damages_wajima.fgb"
    
    # CSVの列名（必要に応じて変更）
    correct_column = 'Correct'
    filename_column = 'filename'
    # ====================================
    
    print("=" * 60)
    print("推論結果CSVから正解画像に対応するFGBファイルを結合")
    print("=" * 60)
    
    # 1. CSVの読み込みとフィルタリング
    print("\n[ステップ1] CSVの読み込みとフィルタリング...")
    fgb_filenames = load_and_filter_csv(csv_path, correct_column, filename_column)
    
    if not fgb_filenames:
        print("処理を終了します。")
        return
    
    # 2. ファイルの検索と収集
    print("\n[ステップ2] FGBファイルの検索...")
    fgb_file_paths = find_fgb_files(fgb_filenames, fgb_directory)
    
    if not fgb_file_paths:
        print("処理を終了します。")
        return
    
    # 3. 地物の統合と出力
    print("\n[ステップ3] 地物の統合と出力...")
    success = merge_vector_layers(fgb_file_paths, output_path)
    
    if success:
        print("\n" + "=" * 60)
        print("処理が正常に完了しました！")
        print(f"出力ファイル: {output_path}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("処理中にエラーが発生しました。")
        print("=" * 60)

main()

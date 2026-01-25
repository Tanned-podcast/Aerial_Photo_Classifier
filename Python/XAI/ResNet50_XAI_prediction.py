"""
ResNet50 画像分類モデルに対する Grad-CAM++ 可視化

要件:
- PyTorch + torchvision.models.resnet50
- 2クラス分類 (0: intact, 1: damaged)
- pytorch-grad-cam を使用して Grad-CAM++ を実装
- 各テスト画像に対して3×2のサブプロット（5枠使用）のPNGを出力
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import warnings
import seaborn as sns
import japanize_matplotlib
sns.set() # seabornの設定を無効化
sns.reset_orig() # seabornの設定をリセットしてmatplotlibのデフォルトに戻す

# matplotlibのスタイルを明示的に設定（seabornの影響を排除）
plt.style.use('default')  # matplotlibのデフォルトスタイルを使用
#seabornのフォントは必ずjapanize_matplotlibのデフォルトのやつに合わせること　でないと文字化け
sns.set(font='IPAexGothic')

# pytorch-grad-cam
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# torchvision
import torchvision.transforms as T
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

warnings.filterwarnings('ignore')


def load_model(model_weight_path: str, num_classes: int = 2, device: torch.device = None, use_pretrained: bool = False) -> torch.nn.Module:
    """
    学習済みResNet50モデルをロードする
    
    Args:
        model_weight_path: 学習済み重みファイルのパス
        num_classes: クラス数（デフォルト: 2）
        device: デバイス
        use_pretrained: 事前学習済み重みを使用するか
    
    Returns:
        ロードされたモデル（evalモード）
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデル構築
    model = resnet50(weights=ResNet50_Weights.DEFAULT if use_pretrained else None)
    
    # 最終層を置き換え
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    if not use_pretrained:
        # 重みをロード
        state_dict = torch.load(model_weight_path, map_location=device)
        model.load_state_dict(state_dict)
    
    # evalモードに設定
    model.eval()
    model.to(device)
    
    return model


def preprocess_image(image_path: str, img_size: Tuple[int, int] = (512, 512)) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    画像を前処理してテンソルに変換する
    
    Args:
        image_path: 画像ファイルのパス
        img_size: リサイズサイズ (H, W)
    
    Returns:
        (preprocessed_tensor, original_image_array, original_size)
    """
    # 画像を読み込み
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)
    
    # 前処理（ImageNet正規化）
    transform = T.Compose([
        T.Resize(img_size, interpolation=Image.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # テンソルに変換
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
    
    # 元画像をnumpy配列に変換（可視化用）
    img_array = np.array(img.resize(img_size, Image.BILINEAR)) / 255.0  # (H, W, 3), [0, 1]
    
    return img_tensor, img_array, original_size


def get_class_probabilities(model: torch.nn.Module, img_tensor: torch.Tensor, 
                           device: torch.device = None) -> Tuple[np.ndarray, int]:
    """
    モデル推論を行い、各クラスの確率と予測クラスを取得する
    
    Args:
        model: 学習済みモデル
        img_tensor: 前処理済み画像テンソル (1, 3, H, W)
        device: デバイス
    
    Returns:
        (probabilities, predicted_class) - probabilitiesは各クラスの確率配列、predicted_classは予測クラスID
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        # 推論
        logits = model(img_tensor)  # (1, num_classes)
        
        # Softmaxを適用
        probs = F.softmax(logits, dim=1)  # (1, num_classes)
        
        # 予測クラス
        pred_class = torch.argmax(probs, dim=1).item()
        
        # numpy配列に変換
        prob_array = probs[0].cpu().numpy()  # (num_classes,)
    
    return prob_array, pred_class


def normalize_map(map_array: np.ndarray) -> np.ndarray:
    """
    マップを[0, 1]範囲に正規化する
    
    Args:
        map_array: 入力マップ
    
    Returns:
        正規化されたマップ
    """
    min_val = map_array.min()
    max_val = map_array.max()
    if max_val - min_val > 1e-8:
        return (map_array - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(map_array)


def visualize_xai_results(
    original_img: np.ndarray,
    cam_intact: np.ndarray,
    cam_damaged: np.ndarray,
    cam_intact_norm: np.ndarray,
    cam_damaged_norm: np.ndarray,
    prob_intact: float,
    prob_damaged: float,
    predicted_class: int,
    true_class: int,
    output_path: str,
    alpha: float = 0.5
):
    """
    3×2のサブプロット（5枠使用）の可視化結果をPNGとして保存する
    
    Args:
        original_img: 元画像 (H, W, 3), [0, 1]
        cam_intact: intactクラスのCAMマップ (H, W)
        cam_damaged: damagedクラスのCAMマップ (H, W)
        cam_intact_norm: 正規化済みintactクラスのCAMマップ (H, W)
        cam_damaged_norm: 正規化済みdamagedクラスのCAMマップ (H, W)
        prob_intact: intactクラスの確率
        prob_damaged: damagedクラスの確率
        predicted_class: 予測クラスID (0: intact, 1: damaged)
        true_class: 正解クラスID (0: intact, 1: damaged)
        output_path: 出力ファイルパス
        alpha: αブレンドの透明度
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    axes = axes.flatten()
    
    fontsize = 24
    labelsize = 18

    # 0. 元画像
    axes[0].imshow(original_img)
    axes[0].set_title('元画像', fontsize=fontsize)
    axes[0].axis('off')

    # 1. intact クラスの Grad-CAM++
    im1 = axes[1].imshow(cam_intact_norm, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('intact クラスの Grad-CAM++', fontsize=fontsize)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).ax.tick_params(labelsize=labelsize)
    
    # 2. intact クラスの CAM + 元画像（overlay）
    cam_intact_overlay = show_cam_on_image(original_img, cam_intact_norm, use_rgb=True, image_weight=1.0-alpha)
    axes[2].imshow(cam_intact_overlay)
    axes[2].set_title('intact クラスの CAM + 元画像', fontsize=fontsize)
    axes[2].axis('off')

    # 3. damaged クラスの Grad-CAM++
    im3 = axes[3].imshow(cam_damaged_norm, cmap='jet', vmin=0, vmax=1)
    axes[3].set_title('damaged クラスの Grad-CAM++', fontsize=fontsize)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04).ax.tick_params(labelsize=labelsize)
    
    # 4. damaged クラスの CAM + 元画像（overlay）
    cam_damaged_overlay = show_cam_on_image(original_img, cam_damaged_norm, use_rgb=True, image_weight=1.0-alpha)
    axes[4].imshow(cam_damaged_overlay)
    axes[4].set_title('damaged クラスの CAM + 元画像', fontsize=fontsize)
    axes[4].axis('off')

    # 5. テキスト情報
    class_names = ['intact', 'damaged']
    pred_class_name = class_names[predicted_class]
    true_class_name = class_names[true_class]
    is_correct = 'Correct' if predicted_class == true_class else 'Incorrect'
    
    text_content = f"""予測確率:
intact: {prob_intact:.2%}
damaged: {prob_damaged:.2%}

予測クラス: {pred_class_name}
正解クラス: {true_class_name}
予測結果: {is_correct}"""
    
    axes[5].text(0.5, 0.5, text_content, 
                ha='center', va='center', fontsize=fontsize, 
                transform=axes[5].transAxes,
                family='monospace')
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_true_class_from_path(image_path: str) -> int:
    """
    画像パスから正解クラスを取得する
    
    Args:
        image_path: 画像ファイルのパス
    
    Returns:
        正解クラスID (0: intact, 1: damaged)
    """
    path_parts = Path(image_path).parts
    # ディレクトリ名からクラスを判定
    if 'intact' in path_parts:
        return 0
    elif 'damaged' in path_parts:
        return 1
    else:
        # デフォルトはintact
        return 0


def process_single_image(
    model: torch.nn.Module,
    cam_method,
    image_path: str,
    output_dir: str,
    target_layer,
    img_size: Tuple[int, int],
    device: torch.device = None,
    alpha: float = 0.5
):
    """
    単一画像に対してXAI可視化を実行する
    
    Args:
        model: 学習済みモデル
        cam_method: CAM手法のインスタンス
        image_path: 画像ファイルのパス
        output_dir: 出力ディレクトリ
        target_layer: ターゲットレイヤ
        img_size: 画像サイズ (H, W)
        device: デバイス
        alpha: αブレンドの透明度
    """
    try:
        # 画像前処理
        img_tensor, img_array, original_size = preprocess_image(image_path, img_size)
        img_tensor = img_tensor.to(device)
        
        # クラス確率と予測クラスを取得
        prob_array, predicted_class = get_class_probabilities(model, img_tensor, device)
        prob_intact = prob_array[0]
        prob_damaged = prob_array[1]
        
        # 正解クラスを取得（ディレクトリ名から）
        true_class = get_true_class_from_path(image_path)
        
        # 両クラスについてGrad-CAM++を計算
        # intact クラス (class_id=0)
        target_intact = [ClassifierOutputTarget(0)]
        grayscale_cam_intact = cam_method(input_tensor=img_tensor, targets=target_intact)
        cam_intact = grayscale_cam_intact[0]  # (H, W)
        cam_intact_norm = normalize_map(cam_intact)
        
        # damaged クラス (class_id=1)
        target_damaged = [ClassifierOutputTarget(1)]
        grayscale_cam_damaged = cam_method(input_tensor=img_tensor, targets=target_damaged)
        cam_damaged = grayscale_cam_damaged[0]  # (H, W)
        cam_damaged_norm = normalize_map(cam_damaged)
        
        # 画像名を取得
        image_name = Path(image_path).stem
        
        # 出力ファイル名
        output_path = os.path.join(output_dir, f"{image_name}.png")
        
        # 可視化
        visualize_xai_results(
            original_img=img_array,
            cam_intact=cam_intact,
            cam_damaged=cam_damaged,
            cam_intact_norm=cam_intact_norm,
            cam_damaged_norm=cam_damaged_norm,
            prob_intact=prob_intact,
            prob_damaged=prob_damaged,
            predicted_class=predicted_class,
            true_class=true_class,
            output_path=output_path,
            alpha=alpha
        )
        
        print(f"✓ {Path(image_path).name}")
        
    except Exception as e:
        print(f"✗ エラー ({Path(image_path).name}): {e}")


def main(
    model_weight_path: str,
    test_image_dir: str,
    output_root_dir: str,
    alpha: float = 0.5,
    img_size: Tuple[int, int] = (224, 224),
    use_pretrained: bool = False,
):
    """
    メイン処理
    
    Args:
        model_weight_path: 学習済み重みファイルのパス
        test_image_dir: テスト画像ディレクトリ（intact/とdamaged/サブディレクトリを含む）
        output_root_dir: 出力先ルートディレクトリ
        alpha: αブレンドの透明度（デフォルト: 0.5）
        img_size: 画像サイズ (H, W)（デフォルト: (224, 224)）
        use_pretrained: 事前学習済み重みを使用するか（デフォルト: False）
    """
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # モデルロード
    print(f"モデルをロード中: {model_weight_path}")
    model = load_model(model_weight_path, num_classes=2, device=device, use_pretrained=use_pretrained)
    print("モデルロード完了")
    
    # ターゲットレイヤの取得（ResNet50の最終畳み込み層）
    target_layer = model.layer4[-1].conv3
    print(f"ターゲットレイヤ: layer4[-1].conv3")
    
    # テスト画像の取得（intact/とdamaged/サブディレクトリから）
    test_image_dir = Path(test_image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    test_images = []
    
    # intact/とdamaged/ディレクトリから画像を取得
    for subdir in ['intact', 'damaged']:
        subdir_path = test_image_dir / subdir
        if subdir_path.exists():
            for ext in image_extensions:
                test_images.extend(subdir_path.glob(f"*{ext}"))
    
    test_images = sorted(test_images)
    
    if len(test_images) == 0:
        print(f"警告: {test_image_dir} に画像が見つかりませんでした")
        return
    
    print(f"テスト画像数: {len(test_images)}")
    
    # 出力ディレクトリの作成
    output_root_dir = Path(output_root_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)
    
    # Grad-CAM++のインスタンス作成
    cam_method = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer],
    )
    
    print(f"\n{'='*60}")
    print(f"Grad-CAM++ で処理中...")
    print(f"{'='*60}")
    
    # 各画像を処理
    for img_path in test_images:
        process_single_image(
            model=model,
            cam_method=cam_method,
            image_path=str(img_path),
            output_dir=str(output_root_dir),
            target_layer=target_layer,
            img_size=img_size,
            device=device,
            alpha=alpha
        )
    
    print(f"\n{'='*60}")
    print("処理完了!")
    print(f"出力先: {output_root_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # パラメータ設定（必要に応じて変更）
    MODEL_WEIGHT_PATH = r"C:\Users\kyohe\Aerial_Photo_Classifier\Sandbox\XAI_test\Weights\model_weights20251212_1758.pth"
    TEST_IMAGE_DIR = r"C:\Users\kyohe\Aerial_Photo_Classifier\Sandbox\XAI_test\Test"  # intact/とdamaged/サブディレクトリを含む
    OUTPUT_ROOT_DIR = r"C:\Users\kyohe\Aerial_Photo_Classifier\Sandbox\XAI_test\Result_XAI"
    ALPHA = 0.4
    IMG_SIZE = (224, 224)
    USE_PRETRAINED = False

    print(f"use_pretrained = {USE_PRETRAINED}")
    
    main(
        model_weight_path=MODEL_WEIGHT_PATH,
        test_image_dir=TEST_IMAGE_DIR,
        output_root_dir=OUTPUT_ROOT_DIR,
        alpha=ALPHA,
        img_size=IMG_SIZE,
        use_pretrained=USE_PRETRAINED
    )

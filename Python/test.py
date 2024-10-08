import glob
import os
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import DataLoader

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

ClassNum=3
Channels=8
batch_size=64
epochs=2

Classes=['被害なし', '崩土', '路肩崩壊']

trainpath=r"C:\Users\AKIZUKI\Desktop\道路空撮画像判定AI\道路画像お試し\訓練"
testpath=r"C:\Users\AKIZUKI\Desktop\道路空撮画像判定AI\道路画像お試し\テスト"

class MyDataset(Dataset):
    def __init__(self, root: str, transforms) -> None:
        super().__init__()
        self.transforms = transforms
        self.data = list(sorted(Path(root).glob("*\*")))
        self.data2 = list(sorted(Path(root+"白黒").glob("*\*")))
        self.Classes =["崩土", "路肩崩壊", "被害なし"]


    # ここで取り出すデータを指定している
    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[index]
        data2 = self.data2[index]

        img1 = Image.open(data)
        img2 = Image.open(data2)

        cat_img = torch.cat((TF.to_tensor(img1), TF.to_tensor(img2)), dim=0)#データの行列のチャンネル数を増やす

        # データの変形 (transforms)
        transformed_img = self.transforms(cat_img)

        label=str(data).split("\\")[-2]
        label = torch.tensor(self.Classes.index(label))

        return transformed_img, label

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self) -> int:
        return len(self.data)


#入力データに施す処理
transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0,0,0,0,0,0,0,0], std=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
])


trainset= MyDataset(root=trainpath, transforms=transforms)
testset= MyDataset(root=testpath, transforms=transforms)

trainloader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True)
testloader = DataLoader(dataset=testset,batch_size=batch_size,shuffle=True)


resnet50 = models.resnet50(pretrained=True)

#modify first layer so it expects 4 input channels; all other parameters unchanged
resnet50.conv1 = torch.nn.Conv2d(Channels,64,kernel_size = (7,7),stride = (2,2), padding = (3,3), bias = False) 
#modifying final layer
resnet50.fc = nn.Linear(2048,ClassNum)

#GPUにネットワークを渡す
resnet50=resnet50.to(device)

#lossfunction&optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)


#trainiterator
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

exit()

#test
inp = torch.rand([1,4,512,512])
resnet50.eval()
resnet50.training = False
out = resnet50(inp) # should evaluate without error
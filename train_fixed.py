from datatset import DROZY_FACE_Dataset
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms, models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 수정된 데이터 전처리 (MobileNetV3에 맞게)
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),  # 2D -> 3D (H, W) -> (C, H, W)
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # 1채널 -> 3채널
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# DROZY_FACE_Dataset이 transform 인자를 받도록 datatset.py에서 수정 필요
train_dataset = DROZY_FACE_Dataset(split_dir="./output_npy/train", transform=transform)
test_dataset = DROZY_FACE_Dataset(split_dir="./output_npy/test", transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# MobileNetV3 모델 불러오기
mobilenet_v3 = models.mobilenet_v3_small(pretrained=True)
# 첫 번째 conv 레이어를 1채널 입력에 맞게 수정
mobilenet_v3.features[0][0] = nn.Conv2d(
    in_channels=3,  # 3채널로 수정
    out_channels=mobilenet_v3.features[0][0].out_channels,
    kernel_size=mobilenet_v3.features[0][0].kernel_size,
    stride=mobilenet_v3.features[0][0].stride,
    padding=mobilenet_v3.features[0][0].padding,
    bias=False
)
mobilenet_v3.classifier[3] = nn.Linear(mobilenet_v3.classifier[3].in_features, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet_v3 = mobilenet_v3.to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mobilenet_v3.parameters(), lr=0.001)

# 훈련 루프
num_epochs = 5
for epoch in range(num_epochs):
    mobilenet_v3.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = mobilenet_v3(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # tqdm 진행바에 현재 loss와 acc 표시
        current_loss = running_loss / total if total > 0 else 0
        current_acc = correct / total * 100 if total > 0 else 0
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

    # 테스트
    mobilenet_v3.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        pbar_test = tqdm(test_dataloader, desc="테스트 진행중")
        for images, labels, _ in pbar_test:
            images = images.to(device)
            labels = labels.to(device)
            outputs = mobilenet_v3(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            current_test_loss = test_loss / test_total if test_total > 0 else 0
            current_test_acc = test_correct / test_total * 100 if test_total > 0 else 0
            pbar_test.set_postfix({'loss': f'{current_test_loss:.4f}', 'acc': f'{current_test_acc:.2f}%'})
        test_epoch_loss = test_loss / test_total
        test_epoch_acc = test_correct / test_total * 100
        print(f"테스트 Loss: {test_epoch_loss:.4f} Acc: {test_epoch_acc:.2f}%")

print("훈련 완료!") 
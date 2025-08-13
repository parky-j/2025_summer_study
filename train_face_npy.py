import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EyeLandmarkDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

        # (1) key_list 생성 (1-1 ~ 14-3)
        self.key_list = [f"{i}-{j}" for i in range(1, 15) for j in range(1, 4)]

        # (2) KSS 값 불러오기 및 매핑
        self.kss_path = "./DROZY/KSS.txt"
        with open(self.kss_path, 'r') as f:
            kss_values = []
            for line in f:
                kss_values.extend(line.strip().split())

        def map_kss_value(val):
            v = int(val)
            if 1 <= v <= 3:
                return 0
            elif 4 <= v <= 6:
                return 1
            else:
                return 2

        self.kss_dict = {}
        for idx, key in enumerate(self.key_list):
            self.kss_dict[key] = map_kss_value(kss_values[idx])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npy_path = self.file_list[idx]
        data = np.load(npy_path)
        # (C, H, W) 형태로 변환 (2D면 1채널로 확장)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        # 라벨 추출: 상위 폴더명에서 key 추출 (예: 1-1, 2-3 등)
        # npy_path: .../train/1-1/clip_0000.npy
        key = os.path.basename(os.path.dirname(npy_path)).replace("_crop", "").replace("_interp_30fps", "")
        label = self.kss_dict[key]
        if self.transform:
            data = self.transform(data)
        return torch.tensor(data, dtype=torch.float32), label

def get_file_lists(base_dir='./DROZY/gaeMuRan/split_eye_landmark_npy_60s'):
    test_npy_files = glob.glob(os.path.join(base_dir, 'test', '*', '*.npy'))
    train_npy_files = glob.glob(os.path.join(base_dir, 'train', '*', '*.npy'))
    return train_npy_files, test_npy_files

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.5], [0.5])
    ])

def get_dataloaders(train_files, test_files, batch_size=32, num_workers=2, transform=None):
    train_dataset = EyeLandmarkDataset(train_files, transform=transform)
    test_dataset = EyeLandmarkDataset(test_files, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def get_model(num_classes=3, pretrained=True, device=None):
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if device is not None:
        model = model.to(device)
    return model

from tqdm import tqdm

def train_and_eval(model, train_loader, test_loader, device, num_epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        loss_list = []
        model.train()
        running_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] 학습", leave=False)
        for data, labels in train_iter:
            data, labels = data.to(device), labels.to(device)
            # (B, C, H*W) -> (B, C, H, W) 변환 필요시
            if data.ndim == 3:
                side = int(np.sqrt(data.shape[2]))
                data = data.view(data.size(0), data.size(1), side, side)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            train_iter.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_list.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(loss_list):.4f}")

        # 검증
        model.eval()
        correct = 0
        total = 0
        val_iter = tqdm(test_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] 검증", leave=False)
        with torch.no_grad():
            for data, labels in val_iter:
                data, labels = data.to(device), labels.to(device)
                if data.ndim == 3:
                    side = int(np.sqrt(data.shape[2]))
                    data = data.view(data.size(0), data.size(1), side, side)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_iter.set_postfix(acc=100 * correct / total if total > 0 else 0)
        acc = 100 * correct / total
        print(f"테스트 정확도: {acc:.2f}%")

if __name__ == "__main__":
    # 파일 리스트 준비
    train_npy_files, test_npy_files = get_file_lists()
    # transform 정의
    transform = None  # npy가 float32이고 (C, H, W)라면 None, 아니면 get_transforms() 사용
    # DataLoader 준비
    train_loader, test_loader = get_dataloaders(train_npy_files, test_npy_files, batch_size=32, num_workers=2, transform=transform)
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 모델 준비
    num_classes = 3
    model = get_model(num_classes=num_classes, pretrained=True, device=device)
    # 학습 및 평가
    train_and_eval(model, train_loader, test_loader, device, num_epochs=10, lr=1e-3)

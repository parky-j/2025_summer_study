import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import numpy as np
import pandas as pd
import random

class DROZY_RRSP_Dataset(Dataset):
    def __init__(self, split_dir, mode='train'):
        self.split_dir = split_dir

        self.sample_path = glob.glob(os.path.join(split_dir, f"*/*.csv"))
        self.kss_path = "./DROZY/KSS.txt"
        self.kss_dict = {}
        key_list = []
        for i in range(1, 15):
            for j in range(1, 4):
                key_list.append(f"{i}-{j}")

        with open(self.kss_path, 'r') as f:
            kss_values = []
            for line in f:
                kss_values.extend(line.strip().split())

        # KSS 값을 1~3은 0, 4~6은 1, 7~9는 3으로 치환
        def map_kss_value(val):
            v = int(val)
            if 1 <= v <= 5:
                return 0
            else:
                return 1
            # elif 7 <= v <= 9:
            #     return 2
            # else:
            #     return v  # 혹시 모를 예외 처리

        for idx, key in enumerate(key_list):
            self.kss_dict[key] = map_kss_value(kss_values[idx])
    
    def __len__(self):
        return len(self.sample_path)
    
    def __getitem__(self, idx):
        sample_path = self.sample_path[idx]
        sample_idx = sample_path.split("\\")[1]

        kss_label = self.kss_dict[sample_idx]

        df = pd.read_csv(sample_path)

        # rrsp_bpm_final 컬럼만 추출하여 리스트로 변환
        # rrsp_bpm_final_upsampled 컬럼이 있으면 그걸, 없으면 rrsp_bpm_final을 사용
        if 'rrsp_bpm_final_upsampled' in df.columns:
            rrsp_bpm_list = df['rrsp_bpm_final_upsampled'].tolist()
        else:
            rrsp_bpm_list = df['rrsp_bpm_final'].tolist()

        # 신호 정규화 포함
        rrsp_bpm_array = np.array(rrsp_bpm_list, dtype=np.float32)
        rrsp_bpm_norm = (rrsp_bpm_array - rrsp_bpm_array.mean()) / (rrsp_bpm_array.std() + 1e-8)
        rrsp_bpm_tensor = torch.tensor(rrsp_bpm_norm, dtype=torch.float32)
        kss_label_tensor = torch.tensor(kss_label, dtype=torch.long)
        return rrsp_bpm_tensor, kss_label_tensor
    
class DROZY_EAR_Dataset(Dataset):
    def __init__(self, split_dir, mode='train', augment=False):
        self.split_dir = split_dir
        self.sample_path = glob.glob(os.path.join(split_dir, f"*/*.csv"))
        self.kss_path = "./DROZY/KSS.txt"
        self.kss_dict = {}
        self.mode = mode
        self.augment = augment if mode == 'train' else False  # train일 때만 증강

        key_list = []
        for i in range(1, 15):
            for j in range(1, 4):
                key_list.append(f"{i}-{j}")

        with open(self.kss_path, 'r') as f:
            kss_values = []
            for line in f:
                kss_values.extend(line.strip().split())

        # KSS 값을 1~3은 0, 4~6은 1, 7~9는 2로 치환
        def map_kss_value(val):
            v = int(val)
            if 1 <= v <= 3:
                return 0
            elif 4 <= v <= 6:
                return 1
            else:
                return 2

        for idx, key in enumerate(key_list):
            self.kss_dict[key] = map_kss_value(kss_values[idx])

    def __len__(self):
        return len(self.sample_path)
    
    def augment_landmark(self, landmark_tensor):
        # 간단한 증강: 가우시안 노이즈 추가, 랜덤 시프트, 랜덤 스케일
        # landmark_tensor: (n_landmarks, seq_len)
        if np.random.rand() < 0.5:
            noise = torch.randn_like(landmark_tensor) * 0.01
            landmark_tensor = landmark_tensor + noise

        if np.random.rand() < 0.5:
            # 랜덤 시프트 (좌우로 시계열 이동)
            shift = np.random.randint(-5, 6)
            landmark_tensor = torch.roll(landmark_tensor, shifts=shift, dims=1)

        if np.random.rand() < 0.5:
            # 랜덤 스케일 (값을 약간 키우거나 줄임)
            scale = 1.0 + (np.random.rand() - 0.5) * 0.1  # 0.95~1.05
            landmark_tensor = landmark_tensor * scale

        return landmark_tensor

    def __getitem__(self, idx):
        # EAR 계산에 사용할 랜드마크 인덱스 (좌/우)
        left_eye_indices = [33, 133, 160, 158, 153, 144]
        right_eye_indices = [263, 362, 387, 385, 380, 373]

        sample_path = self.sample_path[idx]
        sample_idx = sample_path.split("\\")[1]
        kss_label = self.kss_dict[sample_idx]

        df = pd.read_csv(sample_path)

        # 랜드마크 인덱스별로 x, y 좌표 추출
        def get_landmark_xy(df, indices):
            xs = []
            ys = []
            for idx in indices:
                xs.append(df[f"landmark_{idx}_x"].values)
                ys.append(df[f"landmark_{idx}_y"].values)
            # shape: (len(indices), seq_len)
            xs = np.stack(xs, axis=0)
            ys = np.stack(ys, axis=0)
            # (len(indices), seq_len, 2)
            coords = np.stack([xs, ys], axis=-1)
            # (seq_len, len(indices), 2)
            coords = np.transpose(coords, (1, 0, 2))
            return coords  # (seq_len, 6, 2)

        left_eye_coords = get_landmark_xy(df, left_eye_indices)   # (seq_len, 6, 2)
        right_eye_coords = get_landmark_xy(df, right_eye_indices) # (seq_len, 6, 2)

        def euclidean(p1, p2):
            return np.linalg.norm(p1 - p2, axis=-1)  # (seq_len,)

        def compute_ear(eye_coords):
            # eye_coords: (seq_len, 6, 2)
            p1, p2, p3, p4, p5, p6 = [eye_coords[:, i, :] for i in range(6)]
            A = euclidean(p2, p6)
            B = euclidean(p3, p5)
            C = euclidean(p1, p4)
            ear = (A + B) / (2.0 * C + 1e-8)
            return ear  # (seq_len,)

        left_ear = compute_ear(left_eye_coords)    # (seq_len,)
        right_ear = compute_ear(right_eye_coords)  # (seq_len,)

        # 2채널로 구성: (2, seq_len) 형태
        ear_2ch = np.stack([left_ear, right_ear], axis=0)  # (2, seq_len)

        # torch tensor로 변환
        ear_tensor = torch.tensor(ear_2ch, dtype=torch.float32)  # (2, seq_len)
        kss_label_tensor = torch.tensor(kss_label, dtype=torch.long)

        # 증강 적용 (train일 때만)
        if self.augment:
            ear_tensor = self.augment_landmark(ear_tensor)

        return ear_tensor, kss_label_tensor
class DROZY_EYELANDMARK_Dataset(Dataset):
    def __init__(self, split_dir, mode='train', augment=False):
        self.split_dir = split_dir
        self.sample_path = glob.glob(os.path.join(split_dir, f"*/*.csv"))
        self.kss_path = "./DROZY/KSS.txt"
        self.kss_dict = {}
        self.mode = mode
        self.augment = augment if mode == 'train' else False  # train일 때만 증강

        key_list = []
        for i in range(1, 15):
            for j in range(1, 4):
                key_list.append(f"{i}-{j}")

        with open(self.kss_path, 'r') as f:
            kss_values = []
            for line in f:
                kss_values.extend(line.strip().split())

        # KSS 값을 1~3은 0, 4~6은 1, 7~9는 2로 치환
        def map_kss_value(val):
            v = int(val)
            if 1 <= v <= 3:
                return 0
            elif 4 <= v <= 6:
                return 1
            else:
                return 2

        for idx, key in enumerate(key_list):
            self.kss_dict[key] = map_kss_value(kss_values[idx])

    def __len__(self):
        return len(self.sample_path)
    
    def augment_landmark(self, landmark_tensor):
        # 간단한 증강: 가우시안 노이즈 추가, 랜덤 시프트, 랜덤 스케일
        # landmark_tensor: (n_landmarks, seq_len)
        if np.random.rand() < 0.5:
            noise = torch.randn_like(landmark_tensor) * 0.01
            landmark_tensor = landmark_tensor + noise

        if np.random.rand() < 0.5:
            # 랜덤 시프트 (좌우로 시계열 이동)
            shift = np.random.randint(-5, 6)
            landmark_tensor = torch.roll(landmark_tensor, shifts=shift, dims=1)

        if np.random.rand() < 0.5:
            # 랜덤 스케일 (값을 약간 키우거나 줄임)
            scale = 1.0 + (np.random.rand() - 0.5) * 0.1  # 0.95~1.05
            landmark_tensor = landmark_tensor * scale

        return landmark_tensor

    def __getitem__(self, idx):
        sample_path = self.sample_path[idx]
        sample_idx = sample_path.split("\\")[1].replace("_interp_30fps", "")

        kss_label = self.kss_dict[sample_idx]

        df = pd.read_csv(sample_path)

        landmark_cols = [col for col in df.columns if "landmark_" in col]
        landmark_array = np.array(df[landmark_cols].values, dtype=np.float32)
        ######### 학 체크
        if landmark_array.size == 0:
            print(f'size problem임요 쉬봉털{sample_path}')
        if np.isnan(landmark_array).any():
            print(f'nan problem임요 쉬봉털{sample_path}')
        if np.isinf(landmark_array).any():
            print(f'inf problem임요 쉬봉털{sample_path}')
        ######### 학 체크

        # 신호 정규화: 각 랜드마크별로 평균 0, 표준편차 1로 정규화
        # shape: (seq_len, n_landmarks)
        mean = landmark_array.mean(axis=0, keepdims=True)
        std = landmark_array.std(axis=0, keepdims=True) + 1e-8
        landmark_array_norm = (landmark_array - mean) / std

        landmark_tensor = torch.tensor(landmark_array_norm, dtype=torch.float32)
        kss_label_tensor = torch.tensor(kss_label, dtype=torch.long)

        landmark_tensor = landmark_tensor.permute(1, 0)  # (n_landmarks, seq_len)

        # 증강 적용 (train일 때만)
        if self.augment:
            landmark_tensor = self.augment_landmark(landmark_tensor)

        return landmark_tensor, kss_label_tensor, sample_path

class DROZY_FACE_Dataset(Dataset):
    def __init__(self, split_dir, mode='train', transform=None):
        self.split_dir = split_dir
        self.sample_path = glob.glob(os.path.join(split_dir, f"*/*.npy"))
        self.kss_path = "./DROZY/KSS.txt"
        self.kss_dict = {}
        self.mode = mode
        self.transform = transform

        # (1) key_list 생성 (1-1 ~ 14-3)
        key_list = [f"{i}-{j}" for i in range(1, 15) for j in range(1, 4)]

        # (2) KSS 값 불러오기 및 매핑
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
            

        for idx, key in enumerate(key_list):
            self.kss_dict[key] = map_kss_value(kss_values[idx])

    def __len__(self):
        return len(self.sample_path)

        # 시퀀스 단위 동일 증강
    def apply_seq_transform(self, frames, transform):
        seed = np.random.randint(99999)
        out = []
        for img in frames:
            random.seed(seed)
            out.append(transform(img))
        return torch.stack(out)

    def __getitem__(self, idx):
        sample_path = self.sample_path[idx]

        # (3) 운영체제에 독립적인 방식으로 sample_idx 추출
        sample_idx = os.path.basename(os.path.dirname(sample_path)).replace("_crop", "").replace("_interp_30fps", "")
        kss_label = self.kss_dict[sample_idx]

        # (4) npy 파일 로드: shape = (frame, 1, H, W)
        frame_array = np.load(sample_path)

        # frame_array의 shape이 (frame, 1, H, W)이므로, (frame, H, W)로 변환 후 리스트로 만듦
        if frame_array.ndim == 4 and frame_array.shape[1] == 1:
            frame_list = [frame.squeeze(0) for frame in frame_array]  # (1, H, W) -> (H, W)
        else:
            frame_list = [frame for frame in frame_array]

        # albumentations는 images에 numpy array 리스트를 기대함
        if self.transform is not None:
            transformed = self.transform(images=frame_list)
            frame_array = np.stack(transformed['images'])
        else:
            print(f'frame_array shape: {frame_array.shape}')
            frame_array = np.stack(frame_list)

        # (5) (frame, 1, H, W) → (frame, H, W)
        if frame_array.ndim == 4 and frame_array.shape[1] == 1:
            frame_array = frame_array.squeeze(1)
        
        # (frame, H, W) -> (frame, 1, H, W) -> float32 tensor
        frame_tensor = torch.tensor(frame_array, dtype=torch.float32).unsqueeze(1)  # (frame, 1, H, W)

        kss_label_tensor = torch.tensor(kss_label, dtype=torch.long)

        return frame_tensor, kss_label_tensor, sample_path

if __name__ == "__main__":
    import albumentations as A
    transform_video_seg = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomRotate90(p=0.5),
    ])
    dataset = DROZY_FACE_Dataset(split_dir="./DROZY/gaeMuRan/face_cropped_videos_npy_112/train", transform=transform_video_seg)
    print(len(dataset))
    print(dataset[0][0].shape)
    # dataset[0][0]의 nan 값 검사
    import torch

    frame_tensor = dataset[0][0]
    nan_count = torch.isnan(frame_tensor).sum().item()
    print(f"frame_tensor 내 nan 값 개수: {nan_count}")
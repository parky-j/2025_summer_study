import os
import cv2
import numpy as np
from tqdm import tqdm

VIDEO_DIR = 'C:/Users/Genius Park/Desktop/work/2025_Summer_Study/rrsp_drozy/DROZY/gaeMuRan/face_cropped_videos_112'  # 입력: 크롭된 영상 경로
OUTPUT_NPY_DIR = 'C:/Users/Genius Park/Desktop/work/2025_Summer_Study/rrsp_drozy/DROZY/gaeMuRan/face_cropped_videos_npy_112'       # 출력: npy 저장 경로
TARGET_SIZE = (112, 112)
os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)
print(os.listdir(VIDEO_DIR))
for video_file in sorted(os.listdir(VIDEO_DIR)):
    if not video_file.endswith('.mp4'):
        continue

    video_path = os.path.join(VIDEO_DIR, video_file)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or total_frames < fps:
        print(f"⚠️ {video_file} 건너뜀 (fps={fps}, frames={total_frames})")
        cap.release()
        continue

    window_size = int(fps * 3.0)  # 1초 분량
    stride = int(fps * 1.0)  # 0.1초 간격

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, TARGET_SIZE)
        resized = np.expand_dims(resized, axis=0)  # (1, 75, 75)
        frames.append(resized)
    cap.release()

    frames = np.array(frames)  # (N, 1, 75, 75)
    # ImageNet 정규화 추가
    frames = frames.astype(np.float32) / 255.0
    # ImageNet mean/std (grayscale의 경우 R,G,B 동일하게 적용)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # frames shape: (N, 1, 112, 112)
    # 1채널이므로 mean/std의 첫 번째 값만 사용
    frames = (frames - mean[0]) / std[0]

    video_name = os.path.splitext(video_file)[0]
    save_dir = os.path.join(OUTPUT_NPY_DIR, video_name)
    os.makedirs(save_dir, exist_ok=True)

    num_clips = 0
    for start in range(0, len(frames) - window_size + 1, stride):
        clip = frames[start:start + window_size]
        save_path = os.path.join(save_dir, f"clip_{start:04d}.npy")
        np.save(save_path, clip)
        num_clips += 1

    print(f"✅ {video_file} 완료 (fps={fps:.1f}, stride={stride}, clips={num_clips})")

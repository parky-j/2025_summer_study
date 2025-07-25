import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def interpolate_frame(f1, f2, alpha):
    return cv2.addWeighted(f1, alpha, f2, 1 - alpha, 0)

video_dir = r"C:/Users/user/Documents/dataset/DROZY/DROZY/videos_i8"
index_dir = r"C:/Users/user/Documents/dataset/DROZY/DROZY/interpIndices"
output_dir = r"C:/Users/user/Documents/dataset/DROZY/DROZY/interpolated_videos"
os.makedirs(output_dir, exist_ok=True)

video_files = sorted(glob(os.path.join(video_dir, "*.mp4")))

for video_path in tqdm(video_files, desc="전체 영상 처리 중"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    txt_path = os.path.join(index_dir, f"{video_name}.txt")

    if not os.path.exists(txt_path):
        print(f"❌ interpIndices 없음: {video_name}")
        continue

    # 원본 영상 로딩
    cap = cv2.VideoCapture(video_path)
    orig_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig_frames.append(frame)
    cap.release()

    total_frames = len(orig_frames)
    if total_frames < 2:
        print(f"❌ 프레임 너무 적음: {video_name}")
        continue

    h, w = orig_frames[0].shape[:2]

    # interpIndices 읽기
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    if len(lines) != 18000:
        print(f"⚠️ {video_name}의 보간 줄 수가 18000이 아님: {len(lines)}줄")
        continue

    interpolated_frames = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            continue

        idx1, idx2 = int(parts[0]), int(parts[1])
        alpha = float(parts[2])

        if idx1 >= total_frames or idx2 >= total_frames:
            # 유효하지 않은 인덱스 무시
            continue

        f1, f2 = orig_frames[idx1], orig_frames[idx2]
        interp_frame = interpolate_frame(f1, f2, alpha)
        interpolated_frames.append(interp_frame)

    # 영상 저장
    save_path = os.path.join(output_dir, f"{video_name}.mp4")
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
    for frame in interpolated_frames:
        out.write(frame)
    out.release()

    print(f"✅ 저장 완료: {save_path} ({len(interpolated_frames)} frames)")

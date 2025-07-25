import os
import pandas as pd
import numpy as np

CSV_DIR = "./eye_landmarks_csv"  # CSV가 들어 있는 폴더
BUFFER_SECONDS = 10

def get_fps_from_filename(filename):
    base = os.path.basename(filename).split("_")[0]  # '1-2'
    subject, test = map(int, base.split("-"))
    if subject <= 8 and test in [2, 3]:
        return 15
    return 30

def process_rrsp_bpm(file_path, save_dir=None):
    df = pd.read_csv(file_path)

    # if "rrsp_bpm" not in df.columns:
    #     print(f"⚠️ rrsp_bpm 컬럼 없음: {os.path.basename(file_path)}")
    #     return

    # === fps 및 버퍼 길이 계산 ===
    fps = get_fps_from_filename(file_path)
    buffer_frame_count = BUFFER_SECONDS * fps

    # === BPM 처리 ===
    # 'landmark_'가 포함된 컬럼들만 추출하여 처리
    landmark_cols = [col for col in df.columns if "landmark_" in col]
    for col in landmark_cols:
        # 5 이하 값은 0으로 대체 (값이 존재할 때만)
        if df[col].dtype != object:
            col_filtered = df[col].where(df[col] > 5, 0)
        else:
            col_filtered = df[col]

        valid_idx = np.where(col_filtered > 0)[0]
        if len(valid_idx) > 1:
            interpolated = np.interp(np.arange(len(col_filtered)), valid_idx, col_filtered.iloc[valid_idx])
        else:
            interpolated = col_filtered

        # 보간 결과를 새로운 컬럼에 저장
        df[col + "_final"] = interpolated

    df["buffer_frame_count"] = buffer_frame_count  # 동일한 값으로 컬럼 전체 채움

    save_name = os.path.splitext(os.path.basename(file_path))[0] + "_processed.csv"
    save_path = os.path.join(save_dir or os.path.dirname(file_path), save_name)
    df.to_csv(save_path, index=False)
    print(f"✅ {os.path.basename(file_path)} 처리 완료 (fps={fps}, buffer_frames={buffer_frame_count})")

# === 전체 폴더 처리 ===
if __name__ == "__main__":
    for fname in os.listdir(CSV_DIR):
        if fname.endswith("_landmarks.csv"):
            full_path = os.path.join(CSV_DIR, fname)
            process_rrsp_bpm(full_path)

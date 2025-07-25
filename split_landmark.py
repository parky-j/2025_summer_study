import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def interpolate_window_to_30fps(window, original_fps):
    if original_fps == 30:
        return window  # 그대로
    orig_len = window.shape[0]
    new_len = orig_len * 2  # 15fps → 30fps
    orig_t = np.linspace(0, 1, orig_len)
    new_t = np.linspace(0, 1, new_len)
    interp_window = np.empty((new_len, window.shape[1]))
    for i in range(window.shape[1]):
        f = interp1d(orig_t, window[:, i], kind='linear')
        interp_window[:, i] = f(new_t)
    return interp_window

def is_15fps_video(fname):
    for subj in range(1, 9):
        for test in [2, 3]:
            if f"{subj}-{test}" in fname:
                return True
    return False

def split_eye_landmark_csv(csv_dir, output_root, window_sec=20, stride_sec=1, target_fps=30):
    os.makedirs(output_root, exist_ok=True)

    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue

        subject_test = fname.replace("_eye_landmarks.csv", "")
        input_path = os.path.join(csv_dir, fname)
        df = pd.read_csv(input_path)

        landmark_cols = [col for col in df.columns if col.startswith("landmark_")]
        signal = df[landmark_cols].values

        fps = 15 if is_15fps_video(fname) else 30
        window_size = int(window_sec * fps)
        stride_size = int(stride_sec * fps)
        upsample_size = int(window_sec * target_fps)

        subject_dir = os.path.join(output_root, subject_test)
        os.makedirs(subject_dir, exist_ok=True)

        sample_count = 0
        for start in range(300, len(signal) - window_size + 1, stride_size):
            end = start + window_size
            window = signal[start:end, :]

            if fps == target_fps:
                # 컬럼명 유지: landmark_cols 그대로 사용
                sample_df = pd.DataFrame(window, columns=landmark_cols)
                sample_df.insert(0, "start_idx", [start] * window_size)
            else:
                x_old = np.linspace(0, 1, window.shape[0])
                x_new = np.linspace(0, 1, upsample_size)
                upsampled = np.empty((upsample_size, window.shape[1]))
                for i in range(window.shape[1]):
                    interp_func = interp1d(x_old, window[:, i], kind='linear')
                    upsampled[:, i] = interp_func(x_new)
                # 컬럼명 유지: landmark_cols 그대로 사용
                upsampled_df = pd.DataFrame(upsampled, columns=landmark_cols)
                upsampled_df.insert(0, "start_idx", [start] * upsample_size)
                sample_df = upsampled_df

            save_name = f"sample_{start:05d}.csv"
            sample_path = os.path.join(subject_dir, save_name)
            sample_df.to_csv(sample_path, index=False)
            sample_count += 1

        print(f"✅ {fname} → {subject_test}/ : {sample_count}개 저장")

# 예시 실행
if __name__ == "__main__":
    split_eye_landmark_csv(
        csv_dir="G:/DROZY_signals/eye_landmarks_csv",
        output_root="G:/DROZY_signals/split_eye_landmark"
    )
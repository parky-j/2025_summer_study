import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def split_rrsp_all_csvs(csv_dir, output_root, window_sec=480, stride_sec=1, buffer_sec=11, target_fps=30):
    os.makedirs(output_root, exist_ok=True)

    for fname in os.listdir(csv_dir):
        if not fname.endswith("_signals_processed.csv"):
            continue

        input_path = os.path.join(csv_dir, fname)
        df = pd.read_csv(input_path)

        if "rrsp_bpm_final" not in df.columns or "buffer_frame_count" not in df.columns:
            print(f"⚠️ 필요한 컬럼 없음: {fname}")
            continue

        buffer_frame_count = int(df["buffer_frame_count"].iloc[0])
        fps = int(buffer_frame_count / 10)
        start_idx = int(buffer_sec * fps)
        window_size = int(window_sec * fps)
        stride_size = int(stride_sec * fps)
        upsample_size = int(window_sec * target_fps)
        signal = df["rrsp_bpm_final"].values

        # === subject 번호 추출 (예: '1-1' → '1')
        subject_id = fname.replace("_signals_processed.csv", "") 
        subject_dir = os.path.join(output_root, subject_id)
        os.makedirs(subject_dir, exist_ok=True)

        sample_count = 0
        for start in range(start_idx, len(signal) - window_size + 1, stride_size):
            end = start + window_size
            window = signal[start:end]

            if (window == 0).all():
                continue

            if fps == target_fps:
                sample_df = pd.DataFrame({
                    "start_idx": [start] * window_size,
                    "rrsp_bpm_final": window
                })
            else:
                x_old = np.linspace(0, 1, len(window))
                x_new = np.linspace(0, 1, upsample_size)
                interp_func = interp1d(x_old, window, kind='linear')
                upsampled = interp_func(x_new)
                sample_df = pd.DataFrame({
                    "start_idx": [start] * upsample_size,
                    "rrsp_bpm_final_upsampled": upsampled
                })

            save_name = f"sample_{start:05d}.csv"
            sample_path = os.path.join(subject_dir, save_name)
            sample_df.to_csv(sample_path, index=False)
            sample_count += 1

        print(f"✅ {fname} → {subject_id}/ (fps={fps}) : {sample_count}개 저장")

# 예시 실행
if __name__ == "__main__":
    split_rrsp_all_csvs(
        csv_dir="./csv",
        output_root="./split_60"
    )

import os
import cv2
import csv
import numpy as np

from detector.landmark import Detector
from detector.rppg import rPPG
from detector.rrsp import rRSP
from timer import Timer
from utils import draw_signal
from tqdm import tqdm


def draw_result(frame, signal, name, bpm, rect, rect_color=(0, 0, 255)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, rect[0], rect[1], rect_color, 3)

    bpm_w = int(w / 3)
    signal_frame = draw_signal(signal, width=w - bpm_w)
    bpm_frame = np.zeros((150, bpm_w, 3), np.uint8)

    cv2.putText(bpm_frame, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bpm_frame, "%03d" % bpm, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
    return np.vstack((frame, np.hstack((signal_frame, bpm_frame))))


def main():
    video_root = "C:/Users/user/Documents/dataset/DROZY/DROZY/interpolated_videos"
    rrsp_model_path = "C:/Users/user/Documents/orig_rPPG/orig_rPPG/model/onnx_model.onnx"
    output_csv_dir = "C:/Users/user/Documents/dataset/DROZY/DROZY/csv_interpolated"
    os.makedirs(output_csv_dir, exist_ok=True)

    all_videos = sorted([v for v in os.listdir(video_root) if v.endswith(".mp4")])
    print("✅ 전체 파일 리스트:", os.listdir(video_root))
    print("✅ 필터링된 영상 리스트:", all_videos)

    for video_file in tqdm(all_videos, desc="🎬 전체 영상 추론 중"):
        video_name = os.path.splitext(video_file)[0]  # 예: 1-1_interp_30fps
        print(f"🔥 {video_name} 시작")
        video_path = os.path.join(video_root, video_file)
        output_csv_path = os.path.join(output_csv_dir, f"{video_name}_signals.csv")

        if not os.path.exists(video_path):
            print(f"❌ {video_name} 없음, 건너뜀")
            continue

        print(f"🔥 {video_name} 시작")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        detector = Detector()
        rppg = rPPG()
        rrsp = rRSP(rrsp_model_path)
        rrsp.set_fps(fps)

        use_ppg, use_rsp, use_detect = True, True, True
        frame_idx, results = 0, []

        while True:
            Timer.set_time_stamp()
            ret, frame = cap.read()
            if not ret:
                break
            visualize_frame = frame.copy()

            if use_detect:
                ret = detector.process(frame)
            if not ret:
                frame_idx += 1
                continue

            rppg_bpm = 0
            rrsp_bpm = 0

            if use_ppg:
                face_sp, face_ep = detector.get_face_rect()
                rppg_signal = rppg.process(frame, face_sp, face_ep)
                rppg_bpm = rppg.get_bpm()
                visualize_frame = draw_result(visualize_frame, rppg_signal, "rPPG", rppg_bpm, (face_sp, face_ep), (0, 255, 255))

            if use_rsp:
                torso_sp, torso_ep = detector.get_torso_rect()
                rrsp_signal = rrsp.process(frame, torso_sp, torso_ep)
                rrsp_bpm = rrsp.get_bpm()
                visualize_frame = draw_result(visualize_frame, rrsp_signal, "rRSP", rrsp_bpm, (torso_sp, torso_ep), (0, 0, 255))

            cv2.putText(visualize_frame, "%02d fps" % round(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            results.append([frame_idx, rppg_bpm, rrsp_bpm])
            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

        # === CSV 저장 ===
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "rppg_bpm", "rrsp_bpm"])
            writer.writerows(results)

        print(f"✅ 저장 완료: {output_csv_path}")


if __name__ == '__main__':
    # print("🔥 Interpolated 30fps 영상 전체 추론 시작")
    main()

# pyinstaller -D --clean --noconfirm --log-level WARN main.spec

# import cv2
# import numpy as np

# from detector.landmark import Detector
# from detector.rppg import rPPG
# from detector.rrsp import rRSP
# from timer import Timer
# from utils import draw_signal
# import csv

# def draw_result(frame, signal, name, bpm, rect, rect_color=(0, 0, 255)):
#     h, w = frame.shape[:2]
#     cv2.rectangle(frame, rect[0], rect[1], rect_color, 3)

#     bpm_w = int(w / 3)

#     signal_frame = draw_signal(signal, width=w-bpm_w)
#     bpm_frame = np.zeros((150, bpm_w, 3), np.uint8)

#     cv2.putText(bpm_frame, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#     cv2.putText(bpm_frame, "%03d" % bpm, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
#     frame = np.vstack((frame, np.hstack((signal_frame, bpm_frame))))

#     return frame


# def main():
#     # Initialize modules
#     cap = cv2.VideoCapture('C:/Users/user/Documents/dataset/DROZY/DROZY/interpolated_videos/1-1_interp_30fps.mp4')
#     detector = Detector()
#     rppg = rPPG()
#     rrsp = rRSP('./orig_rPPG/model/onnx_model.onnx')

#     # Set flags
#     use_ppg = True
#     use_rsp = True
#     use_detect = True

#     # Set frame
#     # frame_name = 'Remote PPG & RSP'
#     # cv2.namedWindow(frame_name)
    
#     frame_idx = 0
#     results=  []

#     while True:
#         # Set time
#         Timer.set_time_stamp()

#         # Get frame
#         ret, frame = cap.read()
#         if not ret:
#             break
#         visualize_frame = frame.copy()

#         # Calculate landmark
#         if use_detect:
#             ret = detector.process(frame)
#         if not ret:
#             fream_idx += 1
#             continue
#         rppg_bpm = 0
#         rrsp_bpm = 0

    
#         if use_ppg:
#             # Get landmark
#             face_sp, face_ep = detector.get_face_rect()

#             # PPG processing
#             rppg_signal = rppg.process(frame, face_sp, face_ep)
#             rppg_bpm = rppg.get_bpm()

#             # Visualize
#             visualize_frame = draw_result(visualize_frame, rppg_signal, "rPPG", rppg_bpm, (face_sp, face_ep), (0, 255, 255))

#         if use_rsp:
#             # Get torso landmark
#             torso_sp, torso_ep = detector.get_torso_rect()

#             # RSP processing
#             rrsp_signal = rrsp.process(frame, torso_sp, torso_ep)
#             rrsp_bpm = rrsp.get_bpm()

#             # Visualize
#             visualize_frame = draw_result(visualize_frame, rrsp_signal, "rRSP", rrsp_bpm, (torso_sp, torso_ep), (0, 0, 255))

#         # FPS
#         cv2.putText(visualize_frame, "%02d fps" % round(Timer.get_fps()), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#         results.append((frame_idx, rppg_bpm, rrsp_bpm))
#         # Close event
#         # try:
#         #     if cv2.getWindowProperty(frame_name, 0) < 0:
#         #         break
#         # except:
#         #     break

#         cv2.imshow("fdsfasfadfdasfasd", visualize_frame)
#         key = cv2.waitKey(1)

#         if key == 27:
#             break
#         elif key == ord('1'):
#             use_ppg, use_rsp = True, True
#         elif key == ord('2'):
#             use_ppg, use_rsp = True, False
#             rrsp.reset()
#         elif key == ord('3'):
#             use_ppg, use_rsp = False, True
#             rppg.reset()
#         elif key == ord(' '):
#             use_detect = not use_detect

#     cv2.destroyAllWindows()
#     cap.release()

#     # csv_path = "1-1_signals.csv"
#     # with open(csv_path, "w", newline="") as f:
#     #     writer = csv.writer(f)
#     #     writer.writerow(["frame_idx", "rppg_bpm", "rrsp_bpm"])
#     #     writer.writerows(results)


# if __name__ == '__main__':
#     print("씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란씹무란")
#     main()
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

def map_kss_to_label(kss):
    if kss <= 3:
        return 0
    elif kss <= 6:
        return 1
    else:
        return 2

def main():
    # === 설정 ===
    video_root = "C:/Users/user/Documents/dataset/DROZY/DROZY/interpolated_videos"
    rrsp = rRSP('./orig_rPPG/model/onnx_model.onnx')
    output_csv_dir = "C:/Users/user/Documents/dataset/DROZY/DROZY/inter_csv"
    os.makedirs(output_csv_dir, exist_ok=True)

    # === KSS 점수 테이블 ===
    kss_matrix = [
        [3, 6, 7], [3, 7, 6], [2, 3, 4], [4, 8, 9], [3, 7, 8],
        [2, 3, 7], [0, 4, 9], [2, 6, 8], [2, 6, 8], [3, 6, 7],
        [4, 7, 7], [2, 5, 6], [6, 3, 7], [5, 7, 8]
    ]

    # === 모든 subject-test 조합 순회 ===
    for subject_id in range(1, 15):         # 1~14
        for test_id in range(1, 4):         # 1~3
            video_name = f"{subject_id}-{test_id}"
            video_path = os.path.join(video_root, f"{video_name}_interp_30fps.mp4")
            output_csv_path = os.path.join(output_csv_dir, f"{video_name}_signals.csv")

            if not os.path.exists(video_path):
                print(f"❌ {video_name} 없음, 건너뜀")
                continue

            kss_score = kss_matrix[subject_id - 1][test_id - 1]
            label = map_kss_to_label(kss_score)

            print(f"🔥 {video_name} 시작 (label: {label})")

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            detector = Detector()
            rppg = rPPG()
            rrsp = rRSP('./orig_rPPG/model/onnx_model.onnx')
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

                results.append([frame_idx, rppg_bpm, rrsp_bpm, label])
                frame_idx += 1
                cv2.imshow("Frame", visualize_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            # === CSV 저장 ===
            with open(output_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_idx", "rppg_bpm", "rrsp_bpm", "label"])
                writer.writerows(results)

            print(f"✅ 저장 완료: {output_csv_path}")

if __name__ == '__main__':
    print("🔥 DROZY 전체 배치 추론 시작")
    main()

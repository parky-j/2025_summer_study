import cv2
import os
import mediapipe as mp
import numpy as np

# 설정
INPUT_DIR = 'C:/Users/Genius Park/Desktop/work/2025_Summer_Study/rrsp_drozy/DROZY/gaeMuRan/interpolated_videos'  # 원본 영상 경로
OUTPUT_DIR = 'C:/Users/Genius Park/Desktop/work/2025_Summer_Study/rrsp_drozy/DROZY/gaeMuRan/face_cropped_videos_224'  # 얼굴 crop 영상 저장 경로
TARGET_SIZE = (224, 224)  # 얼굴 crop 크기

# MediaPipe 초기화
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def smooth_bbox(bbox_list, window_size=5):
    """bbox_list: [(x1, y1, x2, y2), ...]"""
    bbox_arr = np.array(bbox_list)
    if len(bbox_arr) < window_size:
        return bbox_arr
    smoothed = []
    for i in range(len(bbox_arr)):
        start = max(0, i - window_size // 2)
        end = min(len(bbox_arr), i + window_size // 2 + 1)
        window = bbox_arr[start:end]
        smoothed.append(np.mean(window, axis=0))
    return np.array(smoothed, dtype=int)

# 파일 순회
for subj in range(1, 15):
    for sess in range(1, 4):
        video_name = f"{subj}-{sess}_interp_30fps.mp4"
        video_path = os.path.join(INPUT_DIR, video_name)

        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Processing {video_name} ({fps:.2f} fps)")

        # 얼굴 crop된 프레임을 저장할 리스트
        cropped_frames = []
        bbox_list = []

        # 1차 패스: bbox 좌표 수집
        frames = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                detection = results.detections[0]  # 첫 번째 얼굴만 사용
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1 = int(bbox.xmin * iw)
                y1 = int(bbox.ymin * ih)
                x2 = int((bbox.xmin + bbox.width) * iw)
                y2 = int((bbox.ymin + bbox.height) * ih)
                bbox_list.append([x1, y1, x2, y2])
            else:
                # 이전 bbox가 있으면 마지막 bbox 사용, 없으면 None
                if bbox_list:
                    bbox_list.append(bbox_list[-1])
                else:
                    bbox_list.append([0, 0, frame.shape[1], frame.shape[0]])

        cap.release()

        # bbox smoothing
        if len(bbox_list) > 0:
            smoothed_bboxes = smooth_bbox(bbox_list, window_size=7)
        else:
            smoothed_bboxes = []

        # 2차 패스: crop 및 저장
        for idx, frame in enumerate(frames):
            if idx >= len(smoothed_bboxes):
                continue
            x1, y1, x2, y2 = smoothed_bboxes[idx]
            ih, iw, _ = frame.shape
            # 바운딩 박스 유효성 검사
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(iw, x2)
            y2 = min(ih, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_resized = cv2.resize(face_crop, TARGET_SIZE)
            cropped_frames.append(face_resized)

        # 얼굴 프레임이 있는 경우에만 영상 저장
        if cropped_frames:
            output_path = os.path.join(OUTPUT_DIR, f"{subj}-{sess}_crop.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, TARGET_SIZE)

            for frame in cropped_frames:
                out.write(frame)

            out.release()
            print(f"✅ Saved: {output_path}")

face_detection.close()

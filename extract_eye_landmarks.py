import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

class EyeLandmarkExtractor:
    def __init__(self):
        # MediaPipe Face Mesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 눈, 코, 턱선 랜드마크 인덱스 정의 (MediaPipe Face Mesh 기준)
        # 왼쪽 눈 랜드마크 (468개 포인트 중)
        self.left_eye_landmarks = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382
        ]
        self.right_eye_landmarks = [
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        ]
        self.nose_landmarks = [
            1, 2, 98, 327, 168, 5, 195
        ]
        self.jaw_landmarks = [
            234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400,
            378, 379, 365, 397, 288, 361, 323, 454
        ]


        self.mouth_landmarks = [
            61, 146, 91, 181, 84, 17,     # 상순 (upper lip outer)
            314, 405, 321, 375, 291, 308, # 하순 (alower lip outer)
            78, 95, 88, 178, 87, 14,      # 상순 (upper lip inner)
            317, 402, 318, 324, 308, 415  # 하순 (lower lip inner)
        ]

        self.all_eye_landmarks = list(set(
            self.left_eye_landmarks +
            self.right_eye_landmarks +
            self.nose_landmarks +
            self.jaw_landmarks +
            self.mouth_landmarks
        ))
    
    def extract_landmarks_from_video(self, video_path, output_dir):
        """
        동영상에서 눈 랜드마크를 추출하여 CSV 파일로 저장
        랜드마크가 감지되지 않은 프레임은 이전/이후 프레임의 값으로 보간(interpolate)함
        빠른 처리를 위해 numpy 배열과 벡터화, 미리 할당, tqdm 최소화 등 최적화 적용
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"동영상을 열 수 없습니다: {video_path}")
            return False

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_landmarks = len(self.all_eye_landmarks)
        axes = ['x', 'y', 'z']

        print(f"동영상 처리 중: {os.path.basename(video_path)}")
        print(f"총 프레임 수: {total_frames}, FPS: {fps}")

        # numpy 배열로 미리 할당 (프레임, 랜드마크*3)
        data_arr = np.full((total_frames, n_landmarks * 3), np.nan, dtype=np.float32)
        frame_idx_arr = np.arange(total_frames)
        timestamp_arr = frame_idx_arr / fps

        # 인덱스 매핑 (landmark_idx → 배열 내 위치)
        idx_map = {lm: i for i, lm in enumerate(self.all_eye_landmarks)}

        frame_count = 0
        with tqdm(total=total_frames, desc="프레임 처리", mininterval=1.0) as pbar:
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    for lm in self.all_eye_landmarks:
                        arr_idx = idx_map[lm]
                        if lm < len(face_landmarks.landmark):
                            landmark = face_landmarks.landmark[lm]
                            data_arr[frame_count, arr_idx*3+0] = landmark.x
                            data_arr[frame_count, arr_idx*3+1] = landmark.y
                            data_arr[frame_count, arr_idx*3+2] = landmark.z
                        else:
                            # 이미 np.nan으로 초기화되어 있음
                            pass

                frame_count += 1
                if frame_count % 10 == 0 or frame_count == total_frames:
                    pbar.update(10 if frame_count + 10 <= total_frames else total_frames - frame_count + 10)
            # 남은 프레임 처리
            if frame_count < total_frames:
                pbar.update(total_frames - frame_count)
        cap.release()

        # DataFrame 생성
        columns = []
        for lm in self.all_eye_landmarks:
            for axis in axes:
                columns.append(f'landmark_{lm}_{axis}')
        df = pd.DataFrame(data_arr, columns=columns)
        df.insert(0, 'timestamp', timestamp_arr)
        df.insert(0, 'frame', frame_idx_arr)

        # NaN 보간 (벡터화)
        for col in columns:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')

        # 저장
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_eye_landmarks.csv")
        df.to_csv(output_path, index=False)
        print(f"랜드마크 데이터 저장 완료: {output_path}")
        print(f"총 {len(df)} 프레임의 데이터가 저장되었습니다.")

        return True
    
    def process_all_videos(self, video_dir, output_dir):
        """
        지정된 디렉토리의 모든 동영상을 처리
        """
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 지원하는 비디오 확장자
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        # 비디오 파일 찾기
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
        
        if not video_files:
            print(f"동영상 파일을 찾을 수 없습니다: {video_dir}")
            return
        
        print(f"총 {len(video_files)}개의 동영상 파일을 처리합니다.")
        
        # 각 동영상 처리
        for video_path in video_files:
            try:
                self.extract_landmarks_from_video(video_path, output_dir)
                print("-" * 50)
            except Exception as e:
                print(f"동영상 처리 중 오류 발생: {video_path}")
                print(f"오류 내용: {e}")
                continue
        
        print("모든 동영상 처리 완료!")

def main():
    # MediaPipe Face Mesh 정보 출력
    print("MediaPipe Face Mesh 눈 랜드마크 추출기")
    print("=" * 50)
    
    # 추출기 초기화
    extractor = EyeLandmarkExtractor()
    
    # 눈 랜드마크 정보 출력
    print(f"추출할 눈 랜드마크 개수: {len(extractor.all_eye_landmarks)}")
    print(f"랜드마크 인덱스: {sorted(extractor.all_eye_landmarks)}")
    print()
    # 동영상 디렉토리와 출력 디렉토리 설정
    video_dir = "./DROZY/gaeMuRan/interpolated_videos/"
    output_dir = "./DROZY/gaeMuRan/eye_landmarks_csv/"
    
    # 모든 동영상 처리
    extractor.process_all_videos(video_dir, output_dir)

if __name__ == "__main__":
    import glob
    main()
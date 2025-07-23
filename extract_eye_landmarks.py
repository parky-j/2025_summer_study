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
        
        # 눈 랜드마크 인덱스 정의 (MediaPipe Face Mesh 기준)
        # 왼쪽 눈 랜드마크 (468개 포인트 중)
        self.left_eye_landmarks = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382
        ]
        
        # 오른쪽 눈 랜드마크
        self.right_eye_landmarks = [
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        ]
        
        # 모든 눈 랜드마크 통합
        self.all_eye_landmarks = list(set(self.left_eye_landmarks + self.right_eye_landmarks))
    
    def extract_landmarks_from_video(self, video_path, output_dir):
        """
        동영상에서 눈 랜드마크를 추출하여 CSV 파일로 저장
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"동영상을 열 수 없습니다: {video_path}")
            return False
        
        # 동영상 정보 가져오기
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 결과를 저장할 리스트
        landmarks_data = []
        
        print(f"동영상 처리 중: {os.path.basename(video_path)}")
        print(f"총 프레임 수: {total_frames}, FPS: {fps}")
        
        frame_count = 0
        
        with tqdm(total=total_frames, desc="프레임 처리") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # BGR을 RGB로 변환 (MediaPipe는 RGB 입력 필요)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 얼굴 랜드마크 추출
                results = self.face_mesh.process(rgb_frame)
                
                frame_data = {'frame': frame_count, 'timestamp': frame_count / fps}
                
                if results.multi_face_landmarks:
                    # 첫 번째 얼굴의 랜드마크 사용
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # 눈 랜드마크 좌표 추출
                    for landmark_idx in self.all_eye_landmarks:
                        if landmark_idx < len(face_landmarks.landmark):
                            landmark = face_landmarks.landmark[landmark_idx]
                            frame_data[f'landmark_{landmark_idx}_x'] = landmark.x
                            frame_data[f'landmark_{landmark_idx}_y'] = landmark.y
                            frame_data[f'landmark_{landmark_idx}_z'] = landmark.z
                        else:
                            # 랜드마크가 없는 경우 NaN으로 설정
                            frame_data[f'landmark_{landmark_idx}_x'] = np.nan
                            frame_data[f'landmark_{landmark_idx}_y'] = np.nan
                            frame_data[f'landmark_{landmark_idx}_z'] = np.nan
                else:
                    # 얼굴이 감지되지 않은 경우 모든 랜드마크를 NaN으로 설정
                    for landmark_idx in self.all_eye_landmarks:
                        frame_data[f'landmark_{landmark_idx}_x'] = np.nan
                        frame_data[f'landmark_{landmark_idx}_y'] = np.nan
                        frame_data[f'landmark_{landmark_idx}_z'] = np.nan
                
                landmarks_data.append(frame_data)
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        
        # DataFrame으로 변환하여 CSV 저장
        df = pd.DataFrame(landmarks_data)
        
        # 출력 파일명 생성
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_eye_landmarks.csv")
        
        # CSV 파일로 저장
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
    video_dir = "./DROZY/videos_i8/"
    output_dir = "./eye_landmarks_csv/"
    
    # 모든 동영상 처리
    extractor.process_all_videos(video_dir, output_dir)

if __name__ == "__main__":
    import glob
    main()
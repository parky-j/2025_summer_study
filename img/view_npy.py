import numpy as np

# .npy 파일 경로
npy_path = "./output_npy/1-1_crop/clip_0000.npy"

# 로드
array = np.load(npy_path)

# 배열 정보 출력
print("Shape:", array.shape)        # (fps, 1, 75, 75)
print("Dtype:", array.dtype)        # 보통 uint8 또는 float32
print("Min/Max:", array.min(), array.max())

# 예시: 첫 번째 프레임 보기 (이미지 시각화)
import matplotlib.pyplot as plt

plt.imshow(array[0, 0], cmap='gray')  # (0, 0)은 첫 번째 프레임의 단일 채널
plt.title("첫 번째 프레임")
plt.show()

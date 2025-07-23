import cv2
import numpy as np
import onnxruntime


class rRSP:
    MAX_FPS = 30
    FRAME_BUFFER_SIZE = 5
    MAX_INPUT_SIZE = 200
    BPM_BAND = (10, 50)

    DURATION_MIN_BUFFER = 10
    DURATION_MAX_BUFFER = 17
    DURATION_VISUALIZE = 10
    DURATION_CALCULATE_BPM = 17
    DURATION_ACCUMULATE_BPM = 5

    DURATION_DETREND_KERNEL = 6
    DURATION_LOWPASS_KERNEL = 0.2

    def __init__(self, model_path):
        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)
            print("✅ 모델 세션 생성 성공", flush=True)
        except Exception as e:
            print("❌ 모델 세션 생성 실패:", e, flush=True)
            raise e

        self.fps = self.MAX_FPS
        self.reset()

    def set_fps(self, fps):
        self.fps = fps
        print(f"FPS 설정: {fps}")

    def reset(self):
        self.buffer_frame = []
        self.buffer_raw = [0]
        self.buffer_visualize = []
        self.buffer_bpm = []
        self.bpm = 0

    def process(self, frame, sp, ep):
        fps = self.fps

        self.buffer_frame.append(frame)
        self.buffer_frame = self.buffer_frame[-self.FRAME_BUFFER_SIZE:]
        if len(self.buffer_frame) < self.FRAME_BUFFER_SIZE:
            return [0] * round(self.DURATION_VISUALIZE * fps)

        input1, input2 = self._get_model_input(sp, ep, term=-1)
        val = self._get_respiratory_val(input1, input2)

        self.buffer_raw.append(self.buffer_raw[-1] + val)
        self.buffer_raw = self.buffer_raw[-round(self.DURATION_MAX_BUFFER * fps):]
        if len(self.buffer_raw) < round(self.DURATION_MIN_BUFFER * fps):
            return [0] * round(self.DURATION_VISUALIZE * fps)

        np_signal = np.array(self.buffer_raw[-round(fps * self.DURATION_MAX_BUFFER):])
        detrend = self._detrend_signal(np_signal, fps=fps)
        lowpass = self._lowpass_filter(detrend, fps=fps)

        self.buffer_visualize.append(lowpass[-1])
        self.buffer_visualize = self.buffer_visualize[-round(self.DURATION_VISUALIZE * fps):]

        target_bpm_signal = lowpass[-round(fps * self.DURATION_CALCULATE_BPM):]
        bpm = self._get_bpm(target_bpm_signal, fps)

        self.buffer_bpm.append(bpm)
        self.buffer_bpm = self.buffer_bpm[-round(self.DURATION_ACCUMULATE_BPM * fps):]

        return self.buffer_visualize

    def _get_model_input(self, sp, ep, term=1):
        try:
            input_w = self.MAX_INPUT_SIZE
            input_h = self.MAX_INPUT_SIZE

            curr_frame = self.buffer_frame[-1][sp[1]: ep[1], sp[0]: ep[0], ...]
            prev_frame = self.buffer_frame[-(term + 1)][sp[1]: ep[1], sp[0]: ep[0], ...]

            curr_frame = cv2.resize(curr_frame, dsize=(input_w, input_h)).astype(np.float32) / 255.0
            prev_frame = cv2.resize(prev_frame, dsize=(input_w, input_h)).astype(np.float32) / 255.0

            input_curr = np.expand_dims(np.transpose(curr_frame, (2, 0, 1)), 0)
            input_prev = np.expand_dims(np.transpose(prev_frame, (2, 0, 1)), 0)

            return input_prev, input_curr
        except:
            return None, None

    def _get_respiratory_val(self, input1, input2):
        if input1 is None or input2 is None:
            print("❌ 입력 데이터가 누락. rRSP 계산을 건너뜁니다.", flush=True)
            return 0
        else:
            inputs = {
                self.ort_session.get_inputs()[0].name: input1,
                self.ort_session.get_inputs()[1].name: input2
            }
            output = self.ort_session.run(['output'], inputs)[0][0][0]
            return output.mean()

    def _detrend_signal(self, signal, fps=None):
        if fps is None:
            fps = self.fps
        wsize = min(int(fps * self.DURATION_DETREND_KERNEL), len(signal))
        positive_signal = signal - np.min(signal) + 1
        norm = np.convolve(np.ones(len(positive_signal)), np.ones(wsize), mode='same')
        mean = np.convolve(positive_signal, np.ones(wsize), mode='same') / norm
        detrend = (positive_signal - mean)
        return detrend

    def _lowpass_filter(self, signal, fps=MAX_FPS):
        kernel_size = max(round(fps * self.DURATION_LOWPASS_KERNEL), 3)
        result = np.convolve(signal, np.ones((kernel_size,)) / kernel_size, mode='valid')
        return result

    def _get_bpm(self, signal, fps=MAX_FPS):
        zero_centered = signal - signal.mean()
        padded = np.pad(zero_centered, (0, len(zero_centered) * 9))
        fft_amp = np.abs(np.fft.rfft(padded))
        freq = np.fft.rfftfreq(len(padded), 1 / fps)

        bpm = freq[fft_amp.argmax()] * 60
        if self.BPM_BAND[0] <= bpm <= self.BPM_BAND[1]:
            return bpm
        else:
            return 0

    def get_bpm(self):
        try:
            if len(self.buffer_bpm) > 0:
                sorted_bpm = np.sort(self.buffer_bpm)
                bpm_len = len(sorted_bpm) // 3
                self.bpm = round(sorted_bpm[bpm_len: -bpm_len].mean())
        except:
            pass
        return self.bpm

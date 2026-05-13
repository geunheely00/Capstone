import numpy as np
import sounddevice as sd
import threading
import queue
import time
import math

# =====================================================
# 설정
# =====================================================
SR = 48000
MIC_DISTANCE = 1.0
SPEED_OF_SOUND = 343.0
BLOCK_SIZE = int(SR * 0.1)        # 0.1초씩 처리
WINDOW_SIZE = int(SR * 0.3)       # 0.3초 윈도우로 분석
THRESHOLD = 0.02                  # 트리거 임계값 (피크 기준)
IMPULSE_WINDOW = int(SR * 0.05)   # 임펄스 추출 윈도우 (50ms)

# =====================================================
# 장치 확인
# =====================================================
print("=" * 50)
print("  연결된 오디오 장치 목록")
print("=" * 50)
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        print(f"  [{i}] {d['name']} (입력 {d['max_input_channels']}ch)")

MIC1_DEVICE = int(input("\n마이크1 (왼쪽) 장치 번호: "))
MIC2_DEVICE = int(input("마이크2 (오른쪽) 장치 번호: "))

print(f"\n마이크1: [{MIC1_DEVICE}] {sd.query_devices(MIC1_DEVICE)['name']}")
print(f"마이크2: [{MIC2_DEVICE}] {sd.query_devices(MIC2_DEVICE)['name']}")

# =====================================================
# 연속 녹음 콜백 (큐 기반)
# =====================================================
q1 = queue.Queue()
q2 = queue.Queue()

def callback1(indata, frames, time_info, status):
    q1.put(indata.copy())

def callback2(indata, frames, time_info, status):
    q2.put(indata.copy())

# =====================================================
# 임펄스 도착 시점 정밀 추출
# =====================================================
def find_impulse_onset(signal, threshold_ratio=0.3):
    """신호에서 임펄스 시작 시점(샘플 인덱스) 찾기"""
    abs_sig = np.abs(signal)
    if np.max(abs_sig) < 0.001:
        return None
    
    # 절대값 신호에서 피크 찾기
    peak_idx = np.argmax(abs_sig)
    peak_val = abs_sig[peak_idx]
    
    # 피크 이전에서 threshold_ratio * peak를 넘는 첫 시점 찾기
    threshold = peak_val * threshold_ratio
    onset_idx = peak_idx
    for i in range(peak_idx, max(0, peak_idx - IMPULSE_WINDOW), -1):
        if abs_sig[i] < threshold:
            onset_idx = i + 1
            break
    
    return onset_idx

# =====================================================
# GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
# 일반 교차상관보다 잡음/반향에 강함
# =====================================================
def gcc_phat(sig1, sig2, sr=SR, max_tau=None):
    """GCC-PHAT으로 두 신호 간 시간 차이 계산"""
    n = len(sig1) + len(sig2)
    
    # 길이를 2의 거듭제곱으로 맞춤 (FFT 효율)
    n_fft = 1
    while n_fft < n:
        n_fft *= 2
    
    # FFT
    SIG1 = np.fft.rfft(sig1, n=n_fft)
    SIG2 = np.fft.rfft(sig2, n=n_fft)
    
    # Cross power spectrum
    R = SIG1 * np.conj(SIG2)
    
    # PHAT weighting (위상만 사용 → 잡음/반향에 강함)
    R_phat = R / (np.abs(R) + 1e-15)
    
    # 역 FFT
    cc = np.fft.irfft(R_phat, n=n_fft)
    
    # 물리적으로 가능한 범위만 검색
    if max_tau is None:
        max_tau = MIC_DISTANCE / SPEED_OF_SOUND
    max_shift = int(max_tau * sr) + 5
    
    # 양/음 방향 모두 검색
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift + 1]])
    
    # 최대값 위치
    shift = np.argmax(np.abs(cc)) - max_shift
    
    # 시간 차이
    tau = shift / sr
    
    # 신뢰도 (최대값 / RMS)
    confidence = np.abs(cc[np.argmax(np.abs(cc))]) / (np.sqrt(np.mean(cc**2)) + 1e-15)
    
    return tau, confidence

# =====================================================
# 방향 계산
# =====================================================
def tdoa_to_angle(time_delay):
    ratio = np.clip(time_delay * SPEED_OF_SOUND / MIC_DISTANCE, -1.0, 1.0)
    return math.degrees(math.asin(ratio))

def angle_to_direction(angle):
    if abs(angle) <= 10:
        return "정면"
    elif angle > 10 and angle <= 45:
        return "오른쪽 앞"
    elif angle > 45:
        return "오른쪽"
    elif angle < -10 and angle >= -45:
        return "왼쪽 앞"
    else:
        return "왼쪽"

# =====================================================
# 캘리브레이션 (시작 시점 오프셋 보정)
# =====================================================
print("\n" + "=" * 50)
print("  캘리브레이션")
print("  마이크 정확히 중앙에서 박수 한 번 치세요!")
print("  (3초 후 시작...)")
print("=" * 50)

time.sleep(3)
print("녹음 중... (5초)")

# 스트림 시작
stream1 = sd.InputStream(samplerate=SR, channels=1, device=MIC1_DEVICE,
                         blocksize=BLOCK_SIZE, dtype='float32', callback=callback1)
stream2 = sd.InputStream(samplerate=SR, channels=1, device=MIC2_DEVICE,
                         blocksize=BLOCK_SIZE, dtype='float32', callback=callback2)

stream1.start()
stream2.start()
time.sleep(5)
stream1.stop()
stream2.stop()

# 큐의 모든 데이터 가져오기
buf1 = []
buf2 = []
while not q1.empty():
    buf1.append(q1.get())
while not q2.empty():
    buf2.append(q2.get())

cal_sig1 = np.concatenate(buf1).flatten() if buf1 else np.array([])
cal_sig2 = np.concatenate(buf2).flatten() if buf2 else np.array([])

# 두 신호의 길이를 맞춤
min_len = min(len(cal_sig1), len(cal_sig2))
cal_sig1 = cal_sig1[:min_len]
cal_sig2 = cal_sig2[:min_len]

# 캘리브레이션 박수의 임펄스 위치로 오프셋 계산
onset1 = find_impulse_onset(cal_sig1)
onset2 = find_impulse_onset(cal_sig2)

if onset1 is None or onset2 is None:
    print("[경고] 캘리브레이션 박수를 감지하지 못했습니다. 오프셋 0으로 진행.")
    CALIBRATION_OFFSET = 0
else:
    # 정중앙에서 박수 쳤으니까 두 마이크 도착 시점이 같아야 함
    # 차이가 곧 시스템 오프셋
    CALIBRATION_OFFSET = onset1 - onset2
    offset_ms = CALIBRATION_OFFSET / SR * 1000
    print(f"\n캘리브레이션 완료!")
    print(f"  마이크1 임펄스 위치: {onset1} 샘플")
    print(f"  마이크2 임펄스 위치: {onset2} 샘플")
    print(f"  시스템 오프셋: {CALIBRATION_OFFSET} 샘플 ({offset_ms:.3f}ms)")
    print(f"  → 이 값만큼 보정하여 정확도 향상")

# =====================================================
# 실시간 방향 탐지
# =====================================================
print("\n" + "=" * 50)
print("  실시간 방향 탐지 시작")
print("  배치: 마이크1(왼쪽) ---1m--- 마이크2(오른쪽)")
print("  Ctrl+C로 종료")
print("=" * 50)

# 새 스트림 시작
q1 = queue.Queue()
q2 = queue.Queue()
stream1 = sd.InputStream(samplerate=SR, channels=1, device=MIC1_DEVICE,
                         blocksize=BLOCK_SIZE, dtype='float32', callback=callback1)
stream2 = sd.InputStream(samplerate=SR, channels=1, device=MIC2_DEVICE,
                         blocksize=BLOCK_SIZE, dtype='float32', callback=callback2)
stream1.start()
stream2.start()

buffer1 = np.array([], dtype=np.float32)
buffer2 = np.array([], dtype=np.float32)
count = 0
last_detection_time = 0

try:
    while True:
        # 큐에서 새 데이터 가져오기
        new_data1 = []
        new_data2 = []
        while not q1.empty():
            new_data1.append(q1.get().flatten())
        while not q2.empty():
            new_data2.append(q2.get().flatten())
        
        if new_data1:
            buffer1 = np.concatenate([buffer1] + new_data1)
        if new_data2:
            buffer2 = np.concatenate([buffer2] + new_data2)
        
        # 윈도우 크기보다 크면 분석
        if len(buffer1) >= WINDOW_SIZE and len(buffer2) >= WINDOW_SIZE:
            # 두 버퍼 길이 맞추기
            min_len = min(len(buffer1), len(buffer2))
            sig1 = buffer1[:min_len]
            sig2 = buffer2[:min_len]
            
            # 임계값 이상의 피크가 있는지 확인
            peak1 = np.max(np.abs(sig1))
            peak2 = np.max(np.abs(sig2))
            max_peak = max(peak1, peak2)
            
            # 1초 간격으로만 탐지 (중복 방지)
            current_time = time.time()
            if max_peak > THRESHOLD and (current_time - last_detection_time) > 1.0:
                count += 1
                last_detection_time = current_time
                
                # GCC-PHAT으로 시간 차이 계산
                tau, confidence = gcc_phat(sig1, sig2)
                
                # 캘리브레이션 오프셋 보정
                tau_corrected = tau - (CALIBRATION_OFFSET / SR)
                
                # 방향 계산
                angle = tdoa_to_angle(tau_corrected)
                direction = angle_to_direction(angle)
                
                print(f"\n[{count}] 소리 감지!")
                print(f"  피크: L={peak1:.4f}  R={peak2:.4f}")
                print(f"  원시 시간차: {tau*1000:.3f}ms")
                print(f"  보정 시간차: {tau_corrected*1000:.3f}ms (최대: {MIC_DISTANCE/SPEED_OF_SOUND*1000:.3f}ms)")
                print(f"  방향: {angle:+.1f}도 ({direction})")
                print(f"  신뢰도: {confidence:.2f}")
                
                # 시각적 표시
                bar_pos = int((angle + 90) / 180 * 40)
                bar = list("L" + "-" * 19 + "|" + "-" * 19 + "R")
                if 0 <= bar_pos <= 40:
                    bar[bar_pos] = "★"
                print(f"  {''.join(bar)}")
            
            # 버퍼는 마지막 윈도우만 유지 (메모리 관리)
            buffer1 = buffer1[-WINDOW_SIZE:]
            buffer2 = buffer2[-WINDOW_SIZE:]
        
        time.sleep(0.05)

except KeyboardInterrupt:
    stream1.stop()
    stream2.stop()
    stream1.close()
    stream2.close()
    print("\n\n탐지 종료!")

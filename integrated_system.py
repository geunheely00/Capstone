import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import time

# =====================
# 설정
# =====================
SR = 22050
DURATION = 2.0
N_MELS = 64
MAX_LEN = 128
SOUND_SPEED = 343.0  # 음속 (m/s)

# 마이크 배치 (정삼각형, 간격 1m)
MIC_DISTANCE = 1.0
MIC_POSITIONS = np.array([
    [0.0, 0.0],
    [MIC_DISTANCE, 0.0],
    [MIC_DISTANCE / 2, MIC_DISTANCE * np.sqrt(3) / 2]
])
MIC_CENTER = np.mean(MIC_POSITIONS, axis=0)

# 탐지 조건
TARGET_PROB_THRESHOLD = 60
VOLUME_THRESHOLD = 0.03
PEAK_RATIO_THRESHOLD = 5.0

# =====================
# 모델 로딩
# =====================
model = tf.keras.models.load_model("model/gunshot_model_v4.keras")
print("AI 모델 로딩 완료! (v4 다중분류)")
print(f"마이크 배치: 정삼각형 (간격 {MIC_DISTANCE}m)")
print(f"현재 모드: 마이크 1개 (TDoA는 시뮬레이션)")

# =====================
# 함수 정의
# =====================
def audio_to_mel(audio, sr=SR):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_LEN - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :MAX_LEN]
    return mel_db

def classify_sound(audio):
    """AI로 소리 분류"""
    mel = audio_to_mel(audio)
    mel_input = mel[np.newaxis, ..., np.newaxis]
    prediction = model.predict(mel_input, verbose=0)[0]
    return prediction  # [target, human, background]

def check_impulse_conditions(audio, target_prob):
    """다중 조건으로 임펄스 여부 판별"""
    volume = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    peak_ratio = volume / (rms + 1e-10)

    is_target = (
        target_prob >= TARGET_PROB_THRESHOLD and
        volume >= VOLUME_THRESHOLD and
        peak_ratio >= PEAK_RATIO_THRESHOLD
    )
    return is_target, volume, peak_ratio

def simulate_tdoa(sound_position):
    """
    TDoA 시뮬레이션 (마이크 3개일 때를 가정)
    실제 마이크 3개 연결 시 이 함수를 실제 TDoA 측정으로 교체
    """
    distances = np.array([
        np.linalg.norm(sound_position - mic) for mic in MIC_POSITIONS
    ])
    arrival_times = distances / SOUND_SPEED
    tdoa_21 = arrival_times[1] - arrival_times[0]
    tdoa_31 = arrival_times[2] - arrival_times[0]
    return tdoa_21, tdoa_31

def estimate_position(tdoa_21, tdoa_31):
    """TDoA로 위치 추정 (그리드 서치)"""
    best_pos = None
    min_error = float('inf')

    for x in np.arange(-30, 30, 0.5):
        for y in np.arange(-30, 30, 0.5):
            pos = np.array([x, y])
            d = [np.linalg.norm(pos - mic) for mic in MIC_POSITIONS]
            est_21 = (d[1] - d[0]) / SOUND_SPEED
            est_31 = (d[2] - d[0]) / SOUND_SPEED
            error = (est_21 - tdoa_21)**2 + (est_31 - tdoa_31)**2
            if error < min_error:
                min_error = error
                best_pos = pos

    return best_pos

def calculate_direction_distance(position):
    """추정 위치에서 방향과 거리 계산"""
    direction = position - MIC_CENTER
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    if angle < 0:
        angle += 360
    distance = np.linalg.norm(position - MIC_CENTER)
    return angle, distance

# =====================
# 메인 시스템
# =====================
print("\n" + "=" * 50)
print("  총성 탐지 시스템 v1.0")
print("  AI 분류 + TDoA 거리/방향 추정")
print("=" * 50)
print("\n소리를 내보세요! (종료: Ctrl+C)")
print("탐지 시 시뮬레이션 좌표로 TDoA 계산을 수행합니다.\n")

detection_count = 0

try:
    while True:
        # 1단계: 소리 수신
        audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        volume = np.max(np.abs(audio))
        if volume < 0.02:
            continue  # 조용하면 넘어감

        # 2단계: AI 분류
        prediction = classify_sound(audio)
        target_prob = prediction[0] * 100
        human_prob = prediction[1] * 100
        bg_prob = prediction[2] * 100

        # 3단계: 다중 조건 판별
        is_target, vol, peak_ratio = check_impulse_conditions(audio, target_prob)

        if is_target:
            detection_count += 1
            print(f"\n{'!' * 50}")
            print(f"  [{detection_count}번째 탐지] 대상 소리 감지!")
            print(f"  AI 확률: {target_prob:.1f}% | 볼륨: {vol:.3f} | 피크비율: {peak_ratio:.1f}")

            # 4단계: TDoA 거리/방향 추정 (시뮬레이션)
            # 실제로는 마이크 3개의 시간차를 측정해야 함
            # 지금은 랜덤 위치를 시뮬레이션
            sim_x = np.random.uniform(3, 20)
            sim_y = np.random.uniform(-15, 15)
            sim_position = np.array([sim_x, sim_y])

            print(f"\n  [TDoA 계산 시뮬레이션]")
            print(f"  시뮬레이션 소리 위치: ({sim_x:.1f}, {sim_y:.1f})m")

            tdoa_21, tdoa_31 = simulate_tdoa(sim_position)
            print(f"  TDoA: 마이크2-1={tdoa_21*1000:.4f}ms, 마이크3-1={tdoa_31*1000:.4f}ms")

            estimated = estimate_position(tdoa_21, tdoa_31)
            angle, distance = calculate_direction_distance(estimated)

            print(f"\n  >> 추정 방향: {angle:.1f}도")
            print(f"  >> 추정 거리: {distance:.1f}m")
            print(f"  >> 추정 위치: ({estimated[0]:.1f}, {estimated[1]:.1f})m")
            print(f"  >> 오차: {np.linalg.norm(sim_position - estimated):.2f}m")
            print(f"{'!' * 50}\n")

        else:
            label_names = ["탐지", "사람", "배경"]
            probs = [target_prob, human_prob, bg_prob]
            main = label_names[np.argmax(probs)]
            print(f"   [{main}] 탐지:{target_prob:.1f}% 사람:{human_prob:.1f}% 배경:{bg_prob:.1f}%")

except KeyboardInterrupt:
    print(f"\n\n=== 시스템 종료 ===")
    print(f"총 탐지 횟수: {detection_count}회")
    print("시스템을 종료합니다.")
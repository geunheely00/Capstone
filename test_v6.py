import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf

# v6 2분류 모델 불러오기
model = tf.keras.models.load_model("gunshot_model_v6.keras")
print("모델 로딩 완료! (v6 2분류)")

# v6 모델 학습 시 사용했던 완벽하게 동일한 설정값 (절대 수정 금지)
SR = 48000          # 마이크 샘플링 레이트
DURATION = 0.5      # 0.5초 단위 탐지
N_MELS = 64
HOP_LENGTH = 512
TARGET_LEN = 47     # v6 모델의 입력 가로 길이

def audio_to_mel(audio, sr=SR):
    """v6 학습 코드와 100% 동일한 주파수 대역(100~15000Hz) 필터 적용"""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH,
        n_fft=2048, fmin=100, fmax=15000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # 모델 입력 형태 (64, 47) 정확하게 맞추기
    if mel_db.shape[1] < TARGET_LEN:
        pad = TARGET_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
    else:
        mel_db = mel_db[:, :TARGET_LEN]
        
    return mel_db

print("\n=== 실시간 총성 탐지 시작 (v6) ===")
print("스피커로 총소리, 박수, 책상 치기, 말하기 등을 테스트해 보세요! (종료: Ctrl+C)\n")

try:
    while True:
        # 0.5초 단위로 마이크 입력 받기
        audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        volume = np.max(np.abs(audio))
        if volume < 0.01:
            # 너무 조용한 배경음은 분석하지 않고 패스 (과부하 방지)
            continue

        mel = audio_to_mel(audio)
        mel_input = mel[np.newaxis, ..., np.newaxis] # (1, 64, 47, 1) 로 채널 추가

        # AI 예측 실행
        prediction = model.predict(mel_input, verbose=0)[0]
        
        # v6 라벨: [0: gunshot, 1: not_gunshot]
        gun_prob = prediction[0] * 100
        noise_prob = prediction[1] * 100

        # 다중 조건 판별 (물리적 임펄스 특성 계산)
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        peak_ratio = peak / (rms + 1e-10)  # 높을수록 순간적으로 튀는 소리

        # 조건: 총성 확률 70% 이상 + 볼륨 0.03 이상 + 피크 비율 5배 이상
        is_target = (
            gun_prob >= 80 and
            volume >= 0.01 and
            peak_ratio >= 1.0
        )

        if is_target:
            print(f"🚨 [총성 탐지!] 확률:{gun_prob:.1f}% | 볼륨:{volume:.3f} | 피크비율:{peak_ratio:.1f}")
        else:
            print(f"   [일반 소음] 총성:{gun_prob:.1f}% 비총성:{noise_prob:.1f}% | 볼륨:{volume:.3f} | 피크비율:{peak_ratio:.1f}")

except KeyboardInterrupt:
    print("\n탐지 종료!")

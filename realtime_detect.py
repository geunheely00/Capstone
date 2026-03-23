import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf

# v3 다중분류 모델 불러오기
model = tf.keras.models.load_model("model/gunshot_model_v3.keras")
print("모델 로딩 완료! (v3 다중분류)")

SR = 22050
DURATION = 2.0
N_MELS = 64
MAX_LEN = 128

label_names = ["탐지 대상", "사람 소음", "배경음"]

def audio_to_mel(audio, sr=SR):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_LEN - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :MAX_LEN]
    return mel_db

print("\n=== 실시간 소리 분류 시작 ===")
print("박수, 말하기, 책상 치기 등을 해보세요! (종료: Ctrl+C)\n")

try:
    while True:
        audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        volume = np.max(np.abs(audio))
        if volume < 0.02:
            print("... (조용함)")
            continue

        mel = audio_to_mel(audio)
        mel_input = mel[np.newaxis, ..., np.newaxis]

        prediction = model.predict(mel_input, verbose=0)[0]
        target_prob = prediction[0] * 100
        human_prob = prediction[1] * 100
        bg_prob = prediction[2] * 100

        # 다중 조건 판별
        # 조건1: 탐지 대상 확률이 60% 이상
        # 조건2: 볼륨이 충분히 큼 (임펄스는 순간 음압이 높음)
        # 조건3: 피크 비율 (임펄스는 순간적으로 튀는 소리)
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        peak_ratio = peak / (rms + 1e-10)  # 높을수록 순간적인 소리

        is_target = (
            target_prob >= 60 and
            volume >= 0.03 and
            peak_ratio >= 5.0
        )

        if is_target:
            print(f">> [탐지!] 확률:{target_prob:.1f}% | 볼륨:{volume:.3f} | 피크비율:{peak_ratio:.1f} <<")
        else:
            print(f"   [일반] 탐지:{target_prob:.1f}% 사람:{human_prob:.1f}% 배경:{bg_prob:.1f}% | 볼륨:{volume:.3f} | 피크비율:{peak_ratio:.1f}")

except KeyboardInterrupt:
    print("\n탐지 종료!")
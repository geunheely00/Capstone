import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =====================
# 1단계: 오디오 → 멜 스펙트로그램 변환
# =====================
def audio_to_mel(file_path, sr=22050, n_mels=64, max_len=128):
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=5.0)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 길이 통일 (짧으면 패딩, 길면 자르기)
        if mel_db.shape[1] < max_len:
            pad = max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
        else:
            mel_db = mel_db[:, :max_len]

        return mel_db
    except Exception as e:
        print(f"오류 발생: {file_path} - {e}")
        return None

# =====================
# 2단계: 데이터 로딩
# =====================
print("데이터 로딩 중...")

X = []  # 멜 스펙트로그램 저장
y = []  # 라벨 저장 (1: 임펄스, 0: 비임펄스)

# 임펄스 소리 로딩
impulse_path = "data/impulse"
for filename in os.listdir(impulse_path):
    if filename.endswith(".wav"):
        mel = audio_to_mel(os.path.join(impulse_path, filename))
        if mel is not None:
            X.append(mel)
            y.append(1)

print(f"임펄스 소리 로딩 완료: {sum(y)}개")

# 비임펄스 소리 로딩 (임펄스 개수에 맞춰서 균형 맞추기)
not_impulse_path = "data/not_impulse"
count = 0
max_not_impulse = sum(y) * 2  # 임펄스의 2배만 사용

for filename in os.listdir(not_impulse_path):
    if filename.endswith(".wav") and count < max_not_impulse:
        mel = audio_to_mel(os.path.join(not_impulse_path, filename))
        if mel is not None:
            X.append(mel)
            y.append(0)
            count += 1

print(f"비임펄스 소리 로딩 완료: {count}개")

# 배열 변환
X = np.array(X)
y = np.array(y)

# CNN 입력 형태로 변환 (채널 추가)
X = X[..., np.newaxis]

print(f"전체 데이터: {X.shape[0]}개, 입력 형태: {X.shape[1:]}")

# =====================
# 3단계: 학습/테스트 데이터 분리
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터: {X_train.shape[0]}개, 테스트 데이터: {X_test.shape[0]}개")

# =====================
# 4단계: CNN 모델 설계
# =====================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # 이진 분류
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =====================
# 5단계: 학습 실행
# =====================
print("\n학습 시작!")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# =====================
# 6단계: 결과 평가
# =====================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n테스트 정확도: {test_acc * 100:.1f}%")

# 모델 저장
model.save("model/gunshot_model.keras")
print("모델 저장 완료! (model/gunshot_model.keras)")

# 학습 그래프 저장
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='학습 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.title('정확도')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='학습 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.title('손실')
plt.legend()

plt.savefig("training_result.png")
plt.show()
print("학습 그래프 저장 완료! (training_result.png)")
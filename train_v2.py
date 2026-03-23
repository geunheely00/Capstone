import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =====================
# 1단계: 오디오 → 멜 스펙트로그램
# =====================
SR = 22050
N_MELS = 64
MAX_LEN = 128

def audio_to_mel(file_path):
    try:
        y, _ = librosa.load(file_path, sr=SR, duration=5.0)
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] < MAX_LEN:
            pad = MAX_LEN - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
        else:
            mel_db = mel_db[:, :MAX_LEN]

        return mel_db
    except:
        return None

# =====================
# 2단계: 데이터 증강 (Data Augmentation)
# =====================
def augment_audio(file_path):
    """하나의 파일에서 여러 변형 버전을 만들어서 데이터를 늘림"""
    results = []
    y, _ = librosa.load(file_path, sr=SR, duration=5.0)

    # 원본
    results.append(y)

    # 1) 노이즈 추가
    noise = np.random.normal(0, 0.005, len(y))
    results.append(y + noise)

    # 2) 볼륨 변경 (크게)
    results.append(y * 1.5)

    # 3) 볼륨 변경 (작게)
    results.append(y * 0.7)

    # 4) 시간 이동 (앞부분 잘라내기)
    shift = int(SR * 0.3)
    shifted = np.pad(y[shift:], (0, shift), mode='constant')
    results.append(shifted)

    # 5) 피치 변경
    try:
        y_pitch = librosa.effects.pitch_shift(y, sr=SR, n_steps=2)
        results.append(y_pitch)
    except:
        pass

    return results

def audio_array_to_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < MAX_LEN:
        pad = MAX_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db

# =====================
# 3단계: 데이터 로딩 + 증강
# =====================
print("데이터 로딩 + 증강 중... (시간이 좀 걸려요)")

X = []
y = []

# 임펄스 소리 (증강 적용)
impulse_path = "data/impulse"
for filename in os.listdir(impulse_path):
    if filename.endswith(".wav"):
        filepath = os.path.join(impulse_path, filename)
        try:
            augmented = augment_audio(filepath)
            for audio in augmented:
                mel = audio_array_to_mel(audio.astype(np.float32))
                X.append(mel)
                y.append(1)
        except:
            pass

impulse_total = sum(y)
print(f"임펄스 소리 (증강 후): {impulse_total}개")

# 비임펄스 소리 (증강 없이, 임펄스의 1.5배만 사용)
not_impulse_path = "data/not_impulse"
count = 0
max_not_impulse = int(impulse_total * 1.5)

for filename in os.listdir(not_impulse_path):
    if filename.endswith(".wav") and count < max_not_impulse:
        mel = audio_to_mel(os.path.join(not_impulse_path, filename))
        if mel is not None:
            X.append(mel)
            y.append(0)
            count += 1

print(f"비임펄스 소리: {count}개")

X = np.array(X)
y = np.array(y)
X = X[..., np.newaxis]

print(f"전체 데이터: {X.shape[0]}개")

# =====================
# 4단계: 데이터 분리
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# 5단계: 개선된 CNN 모델
# =====================
model = models.Sequential([
    layers.Input(shape=X.shape[1:]),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 과적합 방지: 검증 손실이 안 줄면 조기 종료
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

model.summary()

# =====================
# 6단계: 학습
# =====================
print("\n학습 시작!")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# =====================
# 7단계: 결과
# =====================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n테스트 정확도: {test_acc * 100:.1f}%")

model.save("model/gunshot_model_v2.keras")
print("모델 저장 완료! (model/gunshot_model_v2.keras)")

# 그래프
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.savefig("training_result_v2.png")
plt.show()
print("그래프 저장 완료!")
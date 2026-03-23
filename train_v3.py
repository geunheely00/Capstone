import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SR = 22050
N_MELS = 64
MAX_LEN = 128

# =====================
# 1단계: 오디오 변환 함수
# =====================
def audio_to_mel(file_path):
    try:
        y, _ = librosa.load(file_path, sr=SR, duration=5.0)
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] < MAX_LEN:
            mel_db = np.pad(mel_db, ((0, 0), (0, MAX_LEN - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :MAX_LEN]
        return mel_db
    except:
        return None

def augment_audio(file_path):
    results = []
    y, _ = librosa.load(file_path, sr=SR, duration=5.0)
    results.append(y)
    results.append(y + np.random.normal(0, 0.005, len(y)))
    results.append(y * 1.5)
    results.append(y * 0.7)
    shift = int(SR * 0.3)
    results.append(np.pad(y[shift:], (0, shift), mode='constant'))
    try:
        results.append(librosa.effects.pitch_shift(y, sr=SR, n_steps=2))
    except:
        pass
    return results

def audio_array_to_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_LEN - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :MAX_LEN]
    return mel_db

# =====================
# 2단계: 데이터 로딩 + 증강
# =====================
print("데이터 로딩 + 증강 중...")

X = []
y = []

# 라벨: 0 = target(탐지대상), 1 = human_noise(사람소음), 2 = background(배경음)
categories = {
    "data_v2/target": 0,
    "data_v2/human_noise": 1,
    "data_v2/background": 2
}

for folder, label in categories.items():
    count = 0
    for filename in os.listdir(folder):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(folder, filename)

        # target은 데이터가 적으니까 증강 적용
        if label == 0:
            try:
                augmented = augment_audio(filepath)
                for audio in augmented:
                    mel = audio_array_to_mel(audio.astype(np.float32))
                    X.append(mel)
                    y.append(label)
                    count += 1
            except:
                pass
        else:
            # human_noise, background는 원본만 사용 (양이 충분)
            # background는 너무 많으니까 제한
            if label == 2 and count >= 400:
                continue
            mel = audio_to_mel(filepath)
            if mel is not None:
                X.append(mel)
                y.append(label)
                count += 1

    name = ["탐지 대상", "사람 소음", "배경음"][label]
    print(f"  {name}: {count}개")

X = np.array(X)
y = np.array(y)
X = X[..., np.newaxis]

print(f"전체 데이터: {X.shape[0]}개")

# =====================
# 3단계: 데이터 분리
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# 4단계: 다중 분류 CNN 모델
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
    layers.Dense(3, activation='softmax')  # 3개 분류!
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',  # 다중 분류용
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

model.summary()

# =====================
# 5단계: 학습
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
# 6단계: 결과
# =====================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n테스트 정확도: {test_acc * 100:.1f}%")

model.save("model/gunshot_model_v3.keras")
print("모델 저장 완료! (model/gunshot_model_v3.keras)")

# 각 카테고리별 정확도 확인
predictions = model.predict(X_test)
pred_labels = np.argmax(predictions, axis=1)

label_names = ["탐지 대상", "사람 소음", "배경음"]
print("\n=== 카테고리별 정확도 ===")
for i, name in enumerate(label_names):
    mask = y_test == i
    if mask.sum() > 0:
        acc = (pred_labels[mask] == i).mean() * 100
        print(f"  {name}: {acc:.1f}% ({mask.sum()}개 중 {(pred_labels[mask] == i).sum()}개 정답)")

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

plt.savefig("training_result_v3.png")
plt.show()
print("그래프 저장 완료!")
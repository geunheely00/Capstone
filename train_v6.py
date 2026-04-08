import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# =====================================================
# 설정
# =====================================================
DATA_DIR = "my_data"
CATEGORIES = ["gunshot", "not_gunshot"]
SR = 48000          # 샘플링 레이트 (마이크 기준)
DURATION = 0.5      # 0.5초 단위로 분석
N_MELS = 64         # 멜 스펙트로그램 밴드 수
HOP_LENGTH = 512

print("=" * 50)
print("  총성 탐지 AI 학습 v6")
print("  데이터: 직접 녹음 데이터만 사용")
print("  분류: 2분류 (총성 / 비총성)")
print("=" * 50)

# =====================================================
# 데이터 로드 + 멜 스펙트로그램 변환
# =====================================================
def extract_segments(file_path, sr=SR, duration=DURATION):
    """하나의 파일에서 여러 개의 0.5초 세그먼트를 추출"""
    try:
        y, _ = librosa.load(file_path, sr=sr)
    except:
        print(f"  [경고] 로드 실패: {file_path}")
        return []
    
    segment_length = int(sr * duration)
    segments = []
    
    # 파일을 0.5초 단위로 분할
    for start in range(0, len(y) - segment_length, segment_length // 2):  # 50% 오버랩
        segment = y[start:start + segment_length]
        if len(segment) == segment_length:
            # RMS가 너무 작으면 (무음) 스킵
            rms = np.sqrt(np.mean(segment ** 2))
            if rms > 0.001:
                segments.append(segment)
    
    return segments

def audio_to_mel(y, sr=SR):
    """오디오를 멜 스펙트로그램으로 변환"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH,
        n_fft=2048, fmin=100, fmax=15000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def augment_audio(y, sr=SR):
    """데이터 증강: 볼륨 변경, 노이즈 추가, 시간 이동"""
    augmented = []
    
    # 1. 볼륨 변경 (0.5배 ~ 1.5배)
    for gain in [0.5, 0.7, 1.3, 1.5]:
        augmented.append(y * gain)
    
    # 2. 노이즈 추가
    for noise_level in [0.002, 0.005]:
        noise = np.random.randn(len(y)) * noise_level
        augmented.append(y + noise)
    
    # 3. 시간 이동 (앞뒤로 살짝)
    for shift in [int(sr * 0.05), int(sr * -0.05)]:
        augmented.append(np.roll(y, shift))
    
    return augmented

# 데이터 로드
X = []
Y = []
file_counts = {}

for label, category in enumerate(CATEGORIES):
    folder = os.path.join(DATA_DIR, category)
    if not os.path.exists(folder):
        print(f"[에러] 폴더를 찾을 수 없습니다: {folder}")
        continue
    
    files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.m4a', '.mp3', '.ogg', '.flac'))]
    file_counts[category] = len(files)
    print(f"\n[{category}] 파일 {len(files)}개 로드 중...")
    
    for fname in files:
        fpath = os.path.join(folder, fname)
        segments = extract_segments(fpath)
        
        for seg in segments:
            # 원본
            mel = audio_to_mel(seg)
            X.append(mel)
            Y.append(label)
            
            # 데이터 증강
            aug_segments = augment_audio(seg)
            for aug_seg in aug_segments:
                aug_seg = np.clip(aug_seg, -1.0, 1.0)
                mel_aug = audio_to_mel(aug_seg)
                X.append(mel_aug)
                Y.append(label)

X = np.array(X)
Y = np.array(Y)

print(f"\n총 데이터 수: {len(X)}")
print(f"  - gunshot: {np.sum(Y == 0)}")
print(f"  - not_gunshot: {np.sum(Y == 1)}")

# CNN 입력 형태로 변환 (채널 추가)
X = X[..., np.newaxis]
print(f"입력 형태: {X.shape}")

# =====================================================
# 학습/테스트 데이터 분리
# =====================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

Y_train_cat = to_categorical(Y_train, num_classes=2)
Y_test_cat = to_categorical(Y_test, num_classes=2)

# 클래스 불균형 보정 (총성 20개 vs 비총성 40개)
weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(Y_train), y=Y_train
)
class_weights = dict(enumerate(weights))
print(f"클래스 가중치: {class_weights}")

# =====================================================
# CNN 모델 구성
# =====================================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =====================================================
# 학습
# =====================================================
early_stop = EarlyStopping(
    monitor='val_accuracy', patience=15,
    restore_best_weights=True, verbose=1
)

print("\n학습 시작...")
history = model.fit(
    X_train, Y_train_cat,
    validation_data=(X_test, Y_test_cat),
    epochs=100,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# =====================================================
# 결과 평가
# =====================================================
test_loss, test_acc = model.evaluate(X_test, Y_test_cat, verbose=0)
print(f"\n{'=' * 50}")
print(f"  최종 테스트 정확도: {test_acc * 100:.1f}%")
print(f"{'=' * 50}")

# 클래스별 정확도
from sklearn.metrics import classification_report
Y_pred = model.predict(X_test)
Y_pred_labels = np.argmax(Y_pred, axis=1)
print("\n클래스별 성능:")
print(classification_report(Y_test, Y_pred_labels, target_names=CATEGORIES))

# =====================================================
# 그래프 저장
# =====================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(history.history['accuracy'], label='학습 정확도')
ax1.plot(history.history['val_accuracy'], label='검증 정확도')
ax1.set_title('정확도 변화')
ax1.set_xlabel('에포크')
ax1.set_ylabel('정확도')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='학습 손실')
ax2.plot(history.history['val_loss'], label='검증 손실')
ax2.set_title('손실 변화')
ax2.set_xlabel('에포크')
ax2.set_ylabel('손실')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(f'총성 탐지 AI v6 - 테스트 정확도: {test_acc * 100:.1f}%', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("train_v6_result.png", dpi=150, bbox_inches='tight')
plt.show()

# =====================================================
# 모델 저장
# =====================================================
model.save("gunshot_model_v6.keras")
print("\n모델 저장 완료: gunshot_model_v6.keras")
print("그래프 저장 완료: train_v6_result.png")

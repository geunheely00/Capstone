import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time

SR = 48000
DURATION = 2  # 녹음 시간 (초)

# 저장 폴더
SAVE_DIR = "my_data"
os.makedirs(f"{SAVE_DIR}/gunshot", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/not_gunshot", exist_ok=True)

# 장치 확인
print("=" * 50)
print("  연결된 입력 장치")
print("=" * 50)
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        print(f"  [{i}] {d['name']}")

DEVICE = int(input("\n마이크 장치 번호: "))
print(f"\n선택: [{DEVICE}] {sd.query_devices(DEVICE)['name']}")

print("\n카테고리 선택:")
print("  1 = gunshot (총성)")
print("  2 = not_gunshot (비총성)")
category = input("번호: ")
folder = f"{SAVE_DIR}/gunshot" if category == "1" else f"{SAVE_DIR}/not_gunshot"

# 기존 파일 수 확인
existing = len([f for f in os.listdir(folder) if f.endswith('.wav')])
count = existing

print(f"\n저장 폴더: {folder} (기존 {existing}개)")
print(f"녹음 시간: {DURATION}초")
print("Enter = 녹음 시작 | q = 종료\n")

while True:
    cmd = input(f"[{count+1}번째] Enter로 녹음 시작... ")
    if cmd.lower() == 'q':
        break
    
    print("  녹음 중...", end="", flush=True)
    audio = sd.rec(int(DURATION * SR), samplerate=SR,
                   channels=1, device=DEVICE, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    
    count += 1
    filename = f"{folder}/rec_{count:03d}.wav"
    sf.write(filename, audio, SR)
    
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    print(f" 저장! → {filename} (볼륨:{rms:.4f}, 피크:{peak:.4f})")

print(f"\n완료! 총 {count}개 저장됨")

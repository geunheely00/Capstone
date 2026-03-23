import sounddevice as sd
import soundfile as sf
import os
import time

SR = 22050
DURATION = 3.0

def record_samples(category, count):
    folder = f"my_data/{category}"
    os.makedirs(folder, exist_ok=True)

    print(f"\n=== [{category}] 녹음 시작 ===")
    print(f"{count}개를 녹음합니다. 각 녹음은 {DURATION}초입니다.\n")

    for i in range(count):
        input(f"  [{i+1}/{count}] Enter를 누르면 녹음 시작...")
        print(f"    녹음 중... ({DURATION}초)")

        audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()

        filename = f"{category}_{i+1:03d}.wav"
        filepath = os.path.join(folder, filename)
        sf.write(filepath, audio, SR)
        print(f"    저장 완료: {filename}")

    print(f"\n[{category}] 녹음 완료! ({count}개)")

print("=== 노트북 마이크 데이터 수집 ===\n")
print("순서대로 진행합니다:")
print("1. 배경음 (아무것도 안 하기)")
print("2. 말하기")
print("3. 박수")
print("4. 책상 치기\n")

record_samples("background", 10)
record_samples("talking", 10)
record_samples("clap", 10)
record_samples("desk", 10)

print("\n=== 전체 녹음 완료! ===")
print("my_data/ 폴더에 40개 파일이 저장되었습니다.")
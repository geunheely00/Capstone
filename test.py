import sounddevice as sd
import soundfile as sf

# 연결된 마이크 목록 출력
print("=== 연결된 마이크 목록 ===")
print(sd.query_devices())

# 3초 녹음 테스트
print("\n3초간 녹음 시작합니다. 뭔가 말해보세요!")
audio = sd.rec(
    int(3 * 44100),
    samplerate=44100,
    channels=1,
    dtype='float32'
)
sd.wait()
sf.write('test.wav', audio, 44100)
print("완료! test.wav 파일이 생성됐습니다.")
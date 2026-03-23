import os
import shutil
import pandas as pd

# 경로 설정 (본인 환경에 맞게 수정)
base_path = "ESC-50-master"
audio_path = os.path.join(base_path, "audio")
meta_path = os.path.join(base_path, "meta", "esc50.csv")

# 분류 폴더 생성
os.makedirs("data/impulse", exist_ok=True)
os.makedirs("data/not_impulse", exist_ok=True)

# 메타 데이터 읽기
df = pd.read_csv(meta_path)

# 임펄스 소리 카테고리 (갑작스럽고 짧은 소리들)
impulse_categories = [
    "gunshot",
    "clapping",
    "fireworks",
    "door_knock",
    "glass_breaking"
]

impulse_count = 0
not_impulse_count = 0

for _, row in df.iterrows():
    filename = row["filename"]
    category = row["category"]
    src = os.path.join(audio_path, filename)

    if not os.path.exists(src):
        continue

    if category in impulse_categories:
        shutil.copy(src, os.path.join("data/impulse", filename))
        impulse_count += 1
    else:
        shutil.copy(src, os.path.join("data/not_impulse", filename))
        not_impulse_count += 1

print(f"분류 완료!")
print(f"임펄스 소리: {impulse_count}개")
print(f"비임펄스 소리: {not_impulse_count}개")
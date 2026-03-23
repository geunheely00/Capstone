import os
import shutil
import pandas as pd

base_path = "ESC-50-master"
audio_path = os.path.join(base_path, "audio")
meta_path = os.path.join(base_path, "meta", "esc50.csv")

# 새 분류 폴더 생성
os.makedirs("data_v2/target", exist_ok=True)
os.makedirs("data_v2/human_noise", exist_ok=True)
os.makedirs("data_v2/background", exist_ok=True)

df = pd.read_csv(meta_path)

# 탐지 대상 (폭발성 소리)
target = ["gunshot", "fireworks", "glass_breaking"]

# 사람이 내는 소음 (무시해야 할 것)
human_noise = ["clapping", "sneezing", "coughing", "crying_baby",
               "laughing", "breathing", "snoring", "drinking_sipping"]

counts = {"target": 0, "human_noise": 0, "background": 0}

for _, row in df.iterrows():
    filename = row["filename"]
    category = row["category"]
    src = os.path.join(audio_path, filename)

    if not os.path.exists(src):
        continue

    if category in target:
        shutil.copy(src, os.path.join("data_v2/target", filename))
        counts["target"] += 1
    elif category in human_noise:
        shutil.copy(src, os.path.join("data_v2/human_noise", filename))
        counts["human_noise"] += 1
    else:
        shutil.copy(src, os.path.join("data_v2/background", filename))
        counts["background"] += 1

print("분류 완료!")
print(f"  탐지 대상 (총성/폭죽/유리): {counts['target']}개")
print(f"  사람 소음 (박수/재채기 등): {counts['human_noise']}개")
print(f"  배경음 (나머지): {counts['background']}개")
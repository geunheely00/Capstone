import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

SOUND_SPEED = 343.0  # m/s

# 마이크 배치 (정삼각형, 간격 1m)
MIC_DISTANCE = 1.0
MIC_POSITIONS = np.array([
    [0.0, 0.0],
    [MIC_DISTANCE, 0.0],
    [MIC_DISTANCE / 2, MIC_DISTANCE * np.sqrt(3) / 2]
])
MIC_CENTER = np.mean(MIC_POSITIONS, axis=0)

# 분석할 샘플링 레이트
sample_rates = [22050, 44100, 48000, 96000]

print("=" * 60)
print("  샘플링 레이트 vs TDoA 정밀도 분석")
print("=" * 60)

# =====================
# 1. 기본 분해능 계산
# =====================
print("\n[1] 기본 시간/거리 분해능")
print("-" * 50)
for sr in sample_rates:
    time_res = 1.0 / sr  # 1샘플 시간 (초)
    dist_res = SOUND_SPEED * time_res  # 거리 분해능 (m)
    print(f"  {sr:>6}Hz: 1샘플 = {time_res*1e6:.1f}μs, 거리 분해능 = {dist_res*100:.2f}cm")

# =====================
# 2. 각도 분해능 분석
# =====================
print("\n[2] 각도별 TDoA 샘플 수 (마이크 간격 1m)")
print("-" * 50)

angles = np.arange(0, 361, 1)  # 0~360도
distances = [5, 10, 20, 50]  # 거리별 분석

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 그래프 1: 각도 vs TDoA 샘플 수 (각 샘플링 레이트별)
ax1 = axes[0][0]
for sr in sample_rates:
    tdoa_samples = []
    for angle in angles:
        rad = np.radians(angle)
        source = MIC_CENTER + np.array([np.cos(rad), np.sin(rad)]) * 10  # 10m 거리
        dists = [np.linalg.norm(source - mic) for mic in MIC_POSITIONS]
        times = np.array(dists) / SOUND_SPEED
        max_tdoa = np.max(times) - np.min(times)
        samples = max_tdoa * sr
        tdoa_samples.append(samples)
    ax1.plot(angles, tdoa_samples, label=f'{sr}Hz')

ax1.set_xlabel('소리 방향 (도)')
ax1.set_ylabel('TDoA 샘플 수')
ax1.set_title('방향별 TDoA 샘플 수 (10m 거리)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 그래프 2: 거리 vs 각도 오차
ax2 = axes[0][1]
dist_range = np.arange(1, 51, 0.5)

for sr in sample_rates:
    angle_errors = []
    for dist in dist_range:
        # 정면(90도)에서의 1샘플 차이가 만드는 각도 오차
        time_res = 1.0 / sr
        # TDoA 변화량 = mic_distance * cos(theta) / sound_speed
        # 1샘플 시간에 해당하는 각도 변화
        delta_angle = np.degrees(np.arcsin(
            min(SOUND_SPEED * time_res / MIC_DISTANCE, 1.0)
        ))
        angle_errors.append(delta_angle)
    ax2.plot(dist_range, angle_errors, label=f'{sr}Hz', linewidth=2)

ax2.set_xlabel('거리 (m)')
ax2.set_ylabel('최소 각도 분해능 (도)')
ax2.set_title('샘플링 레이트별 각도 분해능')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 그래프 3: 거리 vs 위치 오차 (시뮬레이션)
ax3 = axes[1][0]

for sr in sample_rates:
    position_errors = []
    for dist in dist_range:
        # 실제 위치
        true_pos = np.array([dist * 0.5, dist * 0.866])  # 60도 방향
        
        # 실제 TDoA 계산
        true_dists = [np.linalg.norm(true_pos - mic) for mic in MIC_POSITIONS]
        true_times = np.array(true_dists) / SOUND_SPEED
        true_tdoa_21 = true_times[1] - true_times[0]
        true_tdoa_31 = true_times[2] - true_times[0]
        
        # 샘플링에 의한 양자화 (반올림)
        quantized_tdoa_21 = np.round(true_tdoa_21 * sr) / sr
        quantized_tdoa_31 = np.round(true_tdoa_31 * sr) / sr
        
        # 양자화된 TDoA로 위치 추정 (간략화)
        # 양자화 오차
        tdoa_error_21 = abs(quantized_tdoa_21 - true_tdoa_21)
        tdoa_error_31 = abs(quantized_tdoa_31 - true_tdoa_31)
        
        # 거리 오차 근사 (TDoA 오차 * 음속 * 거리 비율)
        dist_error = np.sqrt(
            (tdoa_error_21 * SOUND_SPEED * dist) ** 2 +
            (tdoa_error_31 * SOUND_SPEED * dist) ** 2
        )
        position_errors.append(dist_error)
    
    ax3.plot(dist_range, position_errors, label=f'{sr}Hz', linewidth=2)

ax3.set_xlabel('실제 거리 (m)')
ax3.set_ylabel('위치 추정 오차 (m)')
ax3.set_title('거리별 위치 추정 오차 (양자화 영향)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 그래프 4: 샘플링 레이트 vs 최대 탐지 거리 (목표 오차 이내)
ax4 = axes[1][1]
target_errors = [0.5, 1.0, 2.0, 5.0]  # 목표 오차 (m)
sr_range = np.arange(8000, 100001, 1000)

for target_err in target_errors:
    max_distances = []
    for sr in sr_range:
        time_res = 1.0 / sr
        # 오차가 목표 이내인 최대 거리 근사
        max_dist = target_err / (SOUND_SPEED * time_res * 2)
        max_dist = min(max_dist, 100)
        max_distances.append(max_dist)
    ax4.plot(sr_range / 1000, max_distances, label=f'목표 오차 {target_err}m', linewidth=2)

# 현재 샘플링 레이트 표시
for sr in sample_rates:
    ax4.axvline(x=sr/1000, color='gray', linestyle='--', alpha=0.5)
    ax4.text(sr/1000, 5, f'{sr//1000}k', ha='center', fontsize=9, color='gray')

ax4.set_xlabel('샘플링 레이트 (kHz)')
ax4.set_ylabel('최대 탐지 거리 (m)')
ax4.set_title('목표 오차별 최대 탐지 가능 거리')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 60)

plt.suptitle('샘플링 레이트 vs TDoA 정밀도 분석\n(마이크 간격: 1.0m, 음속: 343 m/s)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("sampling_vs_tdoa.png", dpi=150, bbox_inches='tight')
plt.show()

# =====================
# 3. 요약 테이블
# =====================
print("\n[3] 종합 요약")
print("-" * 70)
print(f"{'샘플링레이트':>10} | {'시간분해능':>10} | {'거리분해능':>10} | {'각도분해능':>10} | {'50m 오차':>10}")
print("-" * 70)
for sr in sample_rates:
    time_res = 1.0 / sr
    dist_res = SOUND_SPEED * time_res
    angle_res = np.degrees(np.arcsin(min(SOUND_SPEED * time_res / MIC_DISTANCE, 1.0)))
    
    # 50m에서의 추정 오차
    true_pos = np.array([25, 43.3])
    true_dists = [np.linalg.norm(true_pos - mic) for mic in MIC_POSITIONS]
    true_times = np.array(true_dists) / SOUND_SPEED
    tdoa_21 = true_times[1] - true_times[0]
    tdoa_31 = true_times[2] - true_times[0]
    q_tdoa_21 = np.round(tdoa_21 * sr) / sr
    q_tdoa_31 = np.round(tdoa_31 * sr) / sr
    err = np.sqrt(
        (abs(q_tdoa_21 - tdoa_21) * SOUND_SPEED * 50) ** 2 +
        (abs(q_tdoa_31 - tdoa_31) * SOUND_SPEED * 50) ** 2
    )
    
    print(f"  {sr:>7}Hz | {time_res*1e6:>8.1f}μs | {dist_res*100:>8.2f}cm | {angle_res:>8.1f}도 | {err:>8.2f}m")

print("\n분석 완료! (sampling_vs_tdoa.png 저장됨)")
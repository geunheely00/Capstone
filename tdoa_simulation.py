import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# =====================
# TDoA 거리/방향 추정 시뮬레이션
# =====================
# 원리: 마이크 3개를 삼각형으로 배치하고,
# 소리가 각 마이크에 도달하는 시간 차이로 소리의 위치를 역산

SOUND_SPEED = 343.0  # 소리 속도 (m/s)

# =====================
# 1단계: 마이크 배치 (정삼각형, 간격 1m)
# =====================
mic_distance = 1.0  # 마이크 간 거리 (미터)

mic_positions = np.array([
    [0.0, 0.0],                                          # 마이크 1 (원점)
    [mic_distance, 0.0],                                  # 마이크 2 (오른쪽)
    [mic_distance / 2, mic_distance * np.sqrt(3) / 2]    # 마이크 3 (위쪽)
])

print("=== 마이크 위치 ===")
for i, pos in enumerate(mic_positions):
    print(f"  마이크 {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}) m")

# =====================
# 2단계: 가상 소리 발생 위치 설정
# =====================
# 실제로는 이 위치를 모르는 상태에서 추정하는 것이 목표
sound_source = np.array([5.0, 8.0])  # 소리 발생 위치 (5m, 8m)

print(f"\n=== 실제 소리 발생 위치 ===")
print(f"  ({sound_source[0]:.1f}, {sound_source[1]:.1f}) m")

# =====================
# 3단계: 각 마이크까지의 도달 시간 계산
# =====================
distances = np.array([
    np.linalg.norm(sound_source - mic) for mic in mic_positions
])
arrival_times = distances / SOUND_SPEED

print(f"\n=== 각 마이크까지 거리 & 도달 시간 ===")
for i in range(3):
    print(f"  마이크 {i+1}: 거리 {distances[i]:.3f}m, 도달 시간 {arrival_times[i]*1000:.4f}ms")

# TDoA 계산 (마이크 1 기준)
tdoa_21 = arrival_times[1] - arrival_times[0]  # 마이크2 - 마이크1
tdoa_31 = arrival_times[2] - arrival_times[0]  # 마이크3 - 마이크1

print(f"\n=== TDoA (시간 차이) ===")
print(f"  마이크2 - 마이크1: {tdoa_21*1000:.4f}ms")
print(f"  마이크3 - 마이크1: {tdoa_31*1000:.4f}ms")

# =====================
# 4단계: TDoA로 위치 추정 (최소제곱법)
# =====================
def estimate_position(mic_positions, tdoa_21, tdoa_31, sound_speed):
    """
    TDoA 기반 위치 추정
    그리드 서치 방식 (단순하지만 직관적)
    """
    best_pos = None
    min_error = float('inf')

    # 탐색 범위 설정
    for x in np.arange(-20, 20, 0.1):
        for y in np.arange(-20, 20, 0.1):
            pos = np.array([x, y])

            # 후보 위치에서 각 마이크까지 거리
            d = [np.linalg.norm(pos - mic) for mic in mic_positions]

            # 예상 TDoA
            est_tdoa_21 = (d[1] - d[0]) / sound_speed
            est_tdoa_31 = (d[2] - d[0]) / sound_speed

            # 실제 TDoA와의 오차
            error = (est_tdoa_21 - tdoa_21)**2 + (est_tdoa_31 - tdoa_31)**2

            if error < min_error:
                min_error = error
                best_pos = pos

    return best_pos

print("\n위치 추정 중... (잠시 기다려주세요)")
estimated_pos = estimate_position(mic_positions, tdoa_21, tdoa_31, SOUND_SPEED)

print(f"\n=== 추정 결과 ===")
print(f"  실제 위치:  ({sound_source[0]:.1f}, {sound_source[1]:.1f}) m")
print(f"  추정 위치:  ({estimated_pos[0]:.1f}, {estimated_pos[1]:.1f}) m")

# 오차 계산
error_distance = np.linalg.norm(sound_source - estimated_pos)
print(f"  오차: {error_distance:.2f} m")

# 방향 계산 (마이크 중심 기준)
mic_center = np.mean(mic_positions, axis=0)
direction_vector = estimated_pos - mic_center
angle = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))

# 거리 계산 (마이크 중심에서 소리까지)
estimated_distance = np.linalg.norm(estimated_pos - mic_center)

print(f"\n=== 최종 결과 ===")
print(f"  추정 방향: {angle:.1f}도")
print(f"  추정 거리: {estimated_distance:.1f}m")

# =====================
# 5단계: 시각화
# =====================
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# 마이크 표시
for i, pos in enumerate(mic_positions):
    ax.plot(pos[0], pos[1], 'bs', markersize=12)
    ax.annotate(f'마이크 {i+1}', (pos[0], pos[1]),
                textcoords="offset points", xytext=(10, 10), fontsize=10)

# 실제 위치
ax.plot(sound_source[0], sound_source[1], 'r*', markersize=20, label='실제 위치')

# 추정 위치
ax.plot(estimated_pos[0], estimated_pos[1], 'gx', markersize=15,
        markeredgewidth=3, label='추정 위치')

# 방향 화살표
ax.annotate('', xy=estimated_pos, xytext=mic_center,
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'TDoA 위치 추정 결과\n추정 거리: {estimated_distance:.1f}m, 방향: {angle:.1f}도, 오차: {error_distance:.2f}m')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("tdoa_simulation_result.png")
plt.show()
print("\n시뮬레이션 완료! (tdoa_simulation_result.png 저장됨)")
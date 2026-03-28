import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# =====================================================
# 이론적 음향 분석 - 마이크 스펙 도출
# =====================================================

# 1. 음원별 SPL (Sound Pressure Level) at 1m
sources = {
    "총성 (소총)":      {"spl_1m": 155, "freq_low": 100,  "freq_high": 15000, "color": "#FF4444", "duration_ms": 3},
    "풍선 터짐":        {"spl_1m": 130, "freq_low": 200,  "freq_high": 12000, "color": "#FF8800", "duration_ms": 5},
    "책 떨어뜨리기":    {"spl_1m": 110, "freq_low": 100,  "freq_high": 8000,  "color": "#44AA44", "duration_ms": 10},
    "박수":             {"spl_1m": 105, "freq_low": 1000, "freq_high": 5000,  "color": "#4488FF", "duration_ms": 15},
    "대화 (참고)":      {"spl_1m": 65,  "freq_low": 100,  "freq_high": 3000,  "color": "#AAAAAA", "duration_ms": 500},
}

# 2. 환경별 소음 수준 (dB SPL)
environments = {
    "야외 조용한 곳": 35,
    "야외 일반": 45,
    "도심 거리": 55,
    "시가전 배경소음 (추정)": 65,
}

# 3. 역제곱 법칙: SPL 감쇠 공식
# SPL(d) = SPL(1m) - 20 * log10(d)
# 추가: 공기 흡수 (고주파에서 약 0.5dB/100m)
def spl_at_distance(spl_1m, distance, freq_high=5000):
    geometric = 20 * np.log10(distance)
    air_absorption = 0.005 * distance * (freq_high / 10000)  # 고주파일수록 흡수 증가
    return spl_1m - geometric - air_absorption

distances = np.arange(1, 101, 0.5)

# =====================================================
# 그래프 1: 거리별 SPL (각 음원)
# =====================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 16))

ax1 = axes[0][0]
for name, info in sources.items():
    spls = [spl_at_distance(info["spl_1m"], d, info["freq_high"]) for d in distances]
    ax1.plot(distances, spls, label=name, color=info["color"], linewidth=2)

# 환경 소음 라인
for env_name, noise in environments.items():
    ax1.axhline(y=noise, linestyle='--', alpha=0.5, color='gray')
    ax1.text(80, noise + 1, env_name, fontsize=8, color='gray')

ax1.axvline(x=50, linestyle=':', color='red', alpha=0.5)
ax1.text(51, 140, '목표 50m', fontsize=10, color='red')
ax1.set_xlabel('거리 (m)')
ax1.set_ylabel('음압 레벨 (dB SPL)')
ax1.set_title('① 거리별 음압 레벨 감쇠')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(20, 170)

# =====================================================
# 그래프 2: 거리별 SNR (시가전 배경소음 기준)
# =====================================================
ax2 = axes[0][1]
noise_floor = 55  # 도심 거리 기준

for name, info in sources.items():
    if name == "대화 (참고)":
        continue
    snrs = [spl_at_distance(info["spl_1m"], d, info["freq_high"]) - noise_floor for d in distances]
    ax2.plot(distances, snrs, label=name, color=info["color"], linewidth=2)

ax2.axhline(y=10, linestyle='--', color='red', linewidth=2, label='최소 SNR 10dB (AI 분류 가능)')
ax2.axhline(y=20, linestyle='--', color='orange', linewidth=1.5, label='권장 SNR 20dB (안정적 분류)')
ax2.axvline(x=50, linestyle=':', color='red', alpha=0.5)
ax2.axhline(y=0, linestyle='-', color='black', linewidth=0.5)
ax2.fill_between(distances, 0, -50, alpha=0.1, color='red', label='SNR < 0 (탐지 불가)')
ax2.set_xlabel('거리 (m)')
ax2.set_ylabel('SNR (dB)')
ax2.set_title('② 거리별 SNR (도심 소음 55dB 기준)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-10, 110)

# =====================================================
# 그래프 3: 음원별 최대 탐지 거리 (환경별)
# =====================================================
ax3 = axes[0][2]
min_snr_values = [10, 15, 20]  # 최소 SNR 기준
source_names = [n for n in sources.keys() if n != "대화 (참고)"]

bar_width = 0.2
x_pos = np.arange(len(source_names))

for i, min_snr in enumerate(min_snr_values):
    max_dists = []
    for name in source_names:
        info = sources[name]
        # 최대 거리 역산: SPL(1m) - 20*log10(d) - noise = min_snr
        # d = 10^((SPL(1m) - noise - min_snr) / 20)
        exponent = (info["spl_1m"] - noise_floor - min_snr) / 20
        max_d = min(10 ** exponent, 200)  # 200m 상한
        max_dists.append(max_d)
    ax3.bar(x_pos + i * bar_width, max_dists, bar_width,
            label=f'SNR ≥ {min_snr}dB', alpha=0.8)

ax3.axhline(y=50, linestyle='--', color='red', linewidth=2)
ax3.text(0, 52, '목표 50m', fontsize=10, color='red', fontweight='bold')
ax3.set_xticks(x_pos + bar_width)
ax3.set_xticklabels(source_names, fontsize=9)
ax3.set_ylabel('최대 탐지 거리 (m)')
ax3.set_title('③ 음원별 최대 탐지 거리 (도심 소음 기준)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# =====================================================
# 그래프 4: 마이크 감도별 최대 탐지 거리
# =====================================================
ax4 = axes[1][0]

# 마이크 감도 (dBV/Pa): -60(저가) ~ -30(고감도)
mic_sensitivities = np.arange(-60, -25, 1)  # dBV/Pa
# 마이크 자체 노이즈 (dB SPL): 감도가 높을수록 보통 자체 노이즈도 낮음
# 일반적 관계 추정
mic_self_noise = np.clip(30 + (mic_sensitivities + 60) * 0.3, 15, 40)

target_sources = ["풍선 터짐", "박수"]
for name in target_sources:
    info = sources[name]
    max_dists_by_sens = []
    for sens, self_noise in zip(mic_sensitivities, mic_self_noise):
        effective_noise = max(noise_floor, self_noise)
        exponent = (info["spl_1m"] - effective_noise - 10) / 20
        max_d = min(10 ** exponent, 200)
        max_dists_by_sens.append(max_d)
    ax4.plot(mic_sensitivities, max_dists_by_sens, label=name,
             color=info["color"], linewidth=2)

ax4.axhline(y=50, linestyle='--', color='red', linewidth=2)
ax4.text(-58, 52, '목표 50m', fontsize=10, color='red')
ax4.axvspan(-44, -38, alpha=0.15, color='green', label='추천 감도 범위')
ax4.set_xlabel('마이크 감도 (dBV/Pa)')
ax4.set_ylabel('최대 탐지 거리 (m)')
ax4.set_title('④ 마이크 감도별 최대 탐지 거리')
ax4.legend()
ax4.grid(True, alpha=0.3)

# =====================================================
# 그래프 5: 음원별 주파수 대역 비교
# =====================================================
ax5 = axes[1][1]

for i, (name, info) in enumerate(sources.items()):
    ax5.barh(i, info["freq_high"] - info["freq_low"],
             left=info["freq_low"], height=0.6,
             color=info["color"], alpha=0.7, label=name)
    ax5.text(info["freq_high"] + 200, i, f'{info["freq_low"]}~{info["freq_high"]}Hz',
             fontsize=9, va='center')

ax5.set_yticks(range(len(sources)))
ax5.set_yticklabels(sources.keys(), fontsize=10)
ax5.set_xlabel('주파수 (Hz)')
ax5.set_title('⑤ 음원별 주파수 대역')
ax5.set_xlim(0, 18000)
ax5.grid(True, alpha=0.3, axis='x')

# 마이크 필요 대역 표시
ax5.axvspan(100, 15000, alpha=0.05, color='cyan')
ax5.text(7000, -0.7, '마이크 필요 대역: 100~15,000Hz', fontsize=10,
         color='teal', fontweight='bold', ha='center')

# =====================================================
# 그래프 6: 종합 마이크 스펙 요약
# =====================================================
ax6 = axes[1][2]
ax6.axis('off')

# 50m에서의 각 음원 SPL 계산
spec_text = "=" * 40 + "\n"
spec_text += "  필요 마이크 스펙 (50m 탐지 기준)\n"
spec_text += "=" * 40 + "\n\n"

spec_text += "【주파수 응답】\n"
spec_text += "  100Hz ~ 15,000Hz 이상\n\n"

spec_text += "【감도】\n"
spec_text += "  -44 ~ -38 dBV/Pa 이상 권장\n"
spec_text += "  (일반 저가 마이크: -60~-50 dBV/Pa)\n\n"

spec_text += "【자체 노이즈】\n"
spec_text += "  30dB SPL 이하 권장\n\n"

spec_text += "【샘플링 레이트】\n"
spec_text += "  48,000Hz (나이퀴스트 + TDoA 정밀도)\n\n"

# 50m 도달 SPL 계산
spec_text += "【50m 도달 음압】\n"
for name, info in sources.items():
    if name == "대화 (참고)":
        continue
    spl_50 = spl_at_distance(info["spl_1m"], 50, info["freq_high"])
    snr_50 = spl_50 - noise_floor
    spec_text += f"  {name}: {spl_50:.1f}dB (SNR: {snr_50:.1f}dB)\n"

ax6.text(0.05, 0.95, spec_text, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('마이크 스펙 도출을 위한 이론적 음향 분석\n(목표 탐지 거리: 50m, 환경: 도심/시가전)',
             fontsize=17, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("mic_spec_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

# =====================================================
# 콘솔 출력: 상세 분석 결과
# =====================================================
print("\n" + "=" * 60)
print("  마이크 스펙 도출 - 이론적 분석 결과")
print("=" * 60)

print("\n[1] 각 음원의 50m 도달 음압")
print("-" * 50)
for name, info in sources.items():
    spl_50 = spl_at_distance(info["spl_1m"], 50, info["freq_high"])
    snr_50 = spl_50 - noise_floor
    status = "✓ 탐지가능" if snr_50 >= 10 else "✗ 탐지어려움"
    print(f"  {name:15s}: 원래 {info['spl_1m']}dB → 50m에서 {spl_50:.1f}dB "
          f"(SNR: {snr_50:.1f}dB) {status}")

print(f"\n[2] 환경 소음 기준: 도심 거리 {noise_floor}dB SPL")

print("\n[3] 음원별 최대 탐지 거리 (SNR ≥ 10dB)")
print("-" * 50)
for name, info in sources.items():
    if name == "대화 (참고)":
        continue
    exponent = (info["spl_1m"] - noise_floor - 10) / 20
    max_d = 10 ** exponent
    print(f"  {name:15s}: 최대 {max_d:.1f}m")

print("\n[4] 필요 마이크 스펙 요약")
print("-" * 50)
print("  주파수 응답:   100Hz ~ 15,000Hz")
print("  감도:          -44 ~ -38 dBV/Pa 이상")
print("  자체 노이즈:   30dB SPL 이하")
print("  샘플링 레이트: 48,000Hz")
print("  마이크 타입:   일렉트릿 콘덴서 (ECM)")

print("\n[5] 마이크 구매 시 확인사항")
print("-" * 50)
print("  1. 감도(Sensitivity)가 -44dBV/Pa 이상인 마이크 선택")
print("  2. 자체 노이즈(Self-noise)가 30dB 이하인지 확인")
print("  3. 주파수 응답이 최소 100Hz~15kHz를 커버하는지 확인")
print("  4. 전원 공급: 플러그인 파워 또는 팬텀 파워 지원 여부")
print("  5. 3개 동일 모델을 구매하여 TDoA 오차 최소화")

print("\n분석 완료! (mic_spec_analysis.png 저장됨)")

# -*- coding:utf-8 -*-
"""합성 데이터로 신호별 quality threshold를 검증한다.

각 신호 특성을 모사한 합성 파형을 생성하고,
정상/노이즈 조건에서 quality score 변화를 확인한다.

사용법:
  source .venv/Scripts/activate
  PYTHONPATH=. python scripts/validate_thresholds_synthetic.py
"""

from __future__ import annotations

import numpy as np

from data.parser._common import segment_quality_score

SR = 100.0  # 100 Hz (전처리 후 저장 SR)
DURATION_S = 60.0
N = int(SR * DURATION_S)
T = np.arange(N) / SR


def make_ecg_like() -> np.ndarray:
    """ECG-like: 주기적 QRS spike + baseline wander."""
    signal = np.zeros(N, dtype=np.float32)
    heart_rate = 70  # bpm
    period_samples = int(SR * 60.0 / heart_rate)
    for i in range(0, N, period_samples):
        # QRS complex: 날카로운 spike (약 40ms 폭)
        qrs_width = int(0.04 * SR)
        for j in range(-qrs_width, qrs_width + 1):
            idx = i + j
            if 0 <= idx < N:
                signal[idx] = 1.5 * np.exp(-0.5 * (j / (qrs_width / 3)) ** 2)
        # T-wave
        for j in range(int(0.2 * SR), int(0.4 * SR)):
            idx = i + j
            if 0 <= idx < N:
                t_phase = (j - 0.2 * SR) / (0.2 * SR)
                signal[idx] += 0.3 * np.sin(np.pi * t_phase)
    # baseline wander
    signal += 0.1 * np.sin(2 * np.pi * 0.15 * T)
    return signal


def make_abp_like() -> np.ndarray:
    """ABP-like: 부드러운 혈압 파형 (systole/diastole)."""
    signal = np.zeros(N, dtype=np.float32)
    heart_rate = 70
    period = SR * 60.0 / heart_rate
    for i in range(N):
        phase = (i % period) / period
        # systolic peak + dicrotic notch
        if phase < 0.35:
            signal[i] = 80 + 40 * np.sin(np.pi * phase / 0.35)
        else:
            decay = (phase - 0.35) / 0.65
            signal[i] = 80 + 40 * np.exp(-3 * decay) * np.cos(2 * np.pi * decay * 0.5)
    return signal


def make_ppg_like() -> np.ndarray:
    """PPG-like: 부드러운 광용적맥파."""
    heart_rate = 70
    freq = heart_rate / 60.0
    signal = 0.5 * np.sin(2 * np.pi * freq * T) + 0.15 * np.sin(4 * np.pi * freq * T)
    signal += 0.02 * np.sin(2 * np.pi * 0.2 * T)  # 호흡
    return signal.astype(np.float32)


def make_co2_like() -> np.ndarray:
    """CO2-like: 느린 capnogram (호흡 주기)."""
    resp_rate = 14  # breaths/min
    freq = resp_rate / 60.0
    signal = np.zeros(N, dtype=np.float32)
    period = int(SR / freq)
    for i in range(N):
        phase = (i % period) / period
        if phase < 0.3:
            # 흡기: 0 유지
            signal[i] = 2.0
        elif phase < 0.8:
            # 호기 plateau: ~38 mmHg
            ramp = (phase - 0.3) / 0.1 if phase < 0.4 else 1.0
            signal[i] = 2.0 + 36.0 * min(ramp, 1.0)
        else:
            # 하강
            descent = (phase - 0.8) / 0.2
            signal[i] = 38.0 - 36.0 * descent
    return signal


def make_eeg_like() -> np.ndarray:
    """EEG-like: alpha (10Hz) + theta (6Hz) + beta (20Hz) 혼합."""
    signal = (
        20.0 * np.sin(2 * np.pi * 10 * T)   # alpha
        + 10.0 * np.sin(2 * np.pi * 6 * T)  # theta
        + 5.0 * np.sin(2 * np.pi * 20 * T)  # beta
        + 2.0 * np.sin(2 * np.pi * 1 * T)   # delta
    )
    return signal.astype(np.float32)


def make_cvp_like() -> np.ndarray:
    """CVP-like: 느린 정맥압 파형 (a, c, v waves)."""
    heart_rate = 70
    period = SR * 60.0 / heart_rate
    signal = np.zeros(N, dtype=np.float32)
    for i in range(N):
        phase = (i % period) / period
        # a-wave, c-wave, v-wave
        signal[i] = (
            8.0
            + 3.0 * np.exp(-((phase - 0.1) ** 2) / 0.005)   # a-wave
            + 1.5 * np.exp(-((phase - 0.25) ** 2) / 0.003)  # c-wave
            + 2.0 * np.exp(-((phase - 0.6) ** 2) / 0.01)    # v-wave
        )
    # 호흡 변동
    signal += 1.0 * np.sin(2 * np.pi * 0.25 * T)
    return signal.astype(np.float32)


def make_awp_like() -> np.ndarray:
    """AWP-like: 환기 주기에 따른 기도압 파형."""
    resp_rate = 14
    freq = resp_rate / 60.0
    period = int(SR / freq)
    signal = np.zeros(N, dtype=np.float32)
    for i in range(N):
        phase = (i % period) / period
        if phase < 0.4:
            # 흡기: 압력 상승
            signal[i] = 5.0 + 15.0 * np.sin(np.pi * phase / 0.4)
        else:
            # 호기: 하강
            descent = (phase - 0.4) / 0.6
            signal[i] = 5.0 + 15.0 * np.exp(-4 * descent)
    return signal


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Gaussian noise를 추가한다 (SNR dB 기준)."""
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(signal)).astype(np.float32) * np.sqrt(noise_power)
    return signal + noise


def add_spikes(signal: np.ndarray, n_spikes: int = 20, amplitude: float = 10.0) -> np.ndarray:
    """랜덤 위치에 임펄스 스파이크를 추가한다."""
    out = signal.copy()
    spike_idx = np.random.randint(0, len(out), n_spikes)
    out[spike_idx] += amplitude * np.random.choice([-1, 1], n_spikes)
    return out


GENERATORS = {
    "ecg": make_ecg_like,
    "abp": make_abp_like,
    "ppg": make_ppg_like,
    "co2": make_co2_like,
    "eeg": make_eeg_like,
    "cvp": make_cvp_like,
    "awp": make_awp_like,
}


def main() -> None:
    np.random.seed(42)

    print("=" * 100)
    print(f"{'Signal':<8} {'Condition':<25} {'flatline':>10} {'clip':>10} {'high_freq':>10} {'pass':>6}")
    print("=" * 100)

    # 추천 threshold (실제 데이터 + 합성 검증 기반)
    recommended_thresholds: dict[str, float] = {}

    for stype, gen_fn in GENERATORS.items():
        clean = gen_fn()

        conditions = [
            ("clean", clean),
            ("noise SNR=20dB", add_noise(clean, 20.0)),
            ("noise SNR=10dB", add_noise(clean, 10.0)),
            ("noise SNR=5dB", add_noise(clean, 5.0)),
            ("noise SNR=0dB", add_noise(clean, 0.0)),
            ("spikes (20)", add_spikes(clean, 20, 10.0)),
            ("spikes (100)", add_spikes(clean, 100, 10.0)),
        ]

        hf_values = []
        for label, sig in conditions:
            score = segment_quality_score(sig)
            hf = score["high_freq_ratio"]
            hf_values.append((label, hf))
            pass_str = "PASS" if score["pass"] else "FAIL"
            print(
                f"{stype:<8} {label:<25} "
                f"{score['flatline_ratio']:>10.4f} "
                f"{score['clip_ratio']:>10.4f} "
                f"{hf:>10.4f} "
                f"{pass_str:>6}"
            )

        # 추천: clean의 2배, 최소 0.5
        clean_hf = hf_values[0][1]
        # SNR=10dB는 "허용 가능한 노이즈" 수준 — 이 값을 상한으로
        snr10_hf = hf_values[2][1]
        # threshold: SNR=10dB 값의 1.2배 (약간의 여유), 최소 clean * 3
        threshold = max(snr10_hf * 1.2, clean_hf * 3.0, 0.5)
        recommended_thresholds[stype] = threshold

        print(f"  → clean={clean_hf:.4f}, SNR10={snr10_hf:.4f}, 추천 threshold={threshold:.2f}")
        print("-" * 100)

    print("\n\n최종 추천 threshold 정리:")
    print("-" * 50)
    for stype in sorted(recommended_thresholds):
        print(f"  {stype:<8}: max_high_freq_ratio = {recommended_thresholds[stype]:.2f}")


if __name__ == "__main__":
    main()

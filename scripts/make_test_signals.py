"""生成合成测试信号，用于验证推理管线连通性。

合成信号只能验证管线跑通与得分响应，不能用于评估识别准确率。
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

SAMPLE_RATE = 16000
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "signals"


def _n_samples(seconds: float) -> int:
    return int(SAMPLE_RATE * seconds)


def sine(freq_hz: float, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(_n_samples(seconds)) / SAMPLE_RATE
    wave = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return (wave * 32767).astype(np.int16)


def white_noise(seconds: float, amplitude: float = 0.2,
                seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wave = rng.uniform(-amplitude, amplitude, _n_samples(seconds))
    return (wave * 32767).astype(np.int16)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(_n_samples(seconds), dtype=np.int16)


def sweep(start_hz: float, end_hz: float, seconds: float) -> np.ndarray:
    t = np.arange(_n_samples(seconds)) / SAMPLE_RATE
    freq = np.linspace(start_hz, end_hz, len(t))
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    return (0.5 * np.sin(phase) * 32767).astype(np.int16)


def pulse_train(rate_hz: float, seconds: float,
                amplitude: float = 0.8) -> np.ndarray:
    """周期性脉冲串，模拟敲击/警报类瞬态信号。"""
    n = _n_samples(seconds)
    wave = np.zeros(n, dtype=np.float32)
    period = max(1, int(SAMPLE_RATE / rate_hz))
    burst = int(0.02 * SAMPLE_RATE)
    for start in range(0, n, period):
        end = min(start + burst, n)
        wave[start:end] = amplitude
    return (wave * 32767).astype(np.int16)


def write_wav(path: str | Path, audio: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), SAMPLE_RATE, audio)


def main() -> int:
    signals = {
        "tone_440hz.wav": sine(440.0, 3.0),
        "tone_1000hz.wav": sine(1000.0, 3.0),
        "sweep_100_8000.wav": sweep(100.0, 8000.0, 3.0),
        "white_noise.wav": white_noise(3.0),
        "silence.wav": silence(3.0),
        "pulses_4hz.wav": pulse_train(4.0, 5.0),
    }
    for name, audio in signals.items():
        write_wav(OUT_DIR / name, audio)
        print(f"已生成 {name}")
    print(f"输出目录: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

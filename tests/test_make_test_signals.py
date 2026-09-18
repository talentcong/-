import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_test_signals import SAMPLE_RATE, sine, silence, white_noise, write_wav


def test_sine_has_expected_length_and_rate():
    tone = sine(440.0, 1.0)
    assert len(tone) == SAMPLE_RATE
    assert tone.dtype == np.int16


def test_sine_frequency_is_correct():
    tone = sine(440.0, 1.0).astype(np.float32) / 32767
    spectrum = np.abs(np.fft.rfft(tone))
    peak_hz = np.fft.rfftfreq(len(tone), 1 / SAMPLE_RATE)[np.argmax(spectrum)]
    assert abs(peak_hz - 440.0) < 2.0


def test_silence_is_all_zero():
    assert np.all(silence(0.5) == 0)


def test_white_noise_is_deterministic_with_seed():
    assert np.array_equal(white_noise(0.5, seed=42), white_noise(0.5, seed=42))


def test_write_wav_creates_readable_file(tmp_path):
    path = tmp_path / "out.wav"
    write_wav(path, sine(440.0, 0.2))
    sr, data = wavfile.read(str(path))
    assert sr == SAMPLE_RATE
    assert len(data) == int(SAMPLE_RATE * 0.2)

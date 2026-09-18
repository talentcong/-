import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audio_io import TARGET_SR, load_audio, normalize_peak, to_mono


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    assert np.allclose(to_mono(stereo), [2.0, 3.0])


def test_to_mono_passes_through_mono():
    mono = np.array([1.0, 2.0], dtype=np.float32)
    assert np.allclose(to_mono(mono), mono)


def test_normalize_peak_scales_to_unit_range():
    assert np.allclose(normalize_peak(np.array([2.0, -4.0], dtype=np.float32)),
                       [0.5, -1.0])


def test_normalize_peak_leaves_silence_untouched():
    assert np.allclose(normalize_peak(np.zeros(4, dtype=np.float32)), np.zeros(4))


def test_load_audio_resamples_44100_to_16000(tmp_path):
    sr_in = 44100
    t = np.arange(sr_in, dtype=np.float32) / sr_in
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    path = tmp_path / "tone44k.wav"
    wavfile.write(path, sr_in, tone)

    audio, sr = load_audio(path)
    assert sr == TARGET_SR
    assert audio.dtype == np.float32
    assert len(audio) == pytest.approx(TARGET_SR, abs=2)
    assert np.max(np.abs(audio)) == pytest.approx(1.0, abs=1e-3)


def test_load_audio_writes_nothing_and_handles_stereo(tmp_path):
    sr_in = 44100
    left = np.full(sr_in, 1000, dtype=np.int16)
    right = np.full(sr_in, -3000, dtype=np.int16)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "stereo.wav"
    wavfile.write(path, sr_in, stereo)

    audio, sr = load_audio(path)
    assert sr == TARGET_SR
    assert audio.ndim == 1

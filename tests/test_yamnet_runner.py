import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from yamnet_runner import N_CLASSES, YamnetRunner

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yamnet.onnx"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="需要先运行 scripts/download_model.py"
)


@pytest.fixture(scope="module")
def runner():
    return YamnetRunner(MODEL_PATH)


def test_score_returns_one_row_per_frame(runner):
    wave = np.zeros(16000, dtype=np.float32)
    scores = runner.score(wave)
    assert scores.ndim == 2
    assert scores.shape[1] == N_CLASSES
    assert scores.shape[0] >= 1


def test_score_values_look_like_probabilities(runner):
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 0.1, 16000 * 2).astype(np.float32)
    scores = runner.score(wave)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_empty_audio_raises(runner):
    with pytest.raises(ValueError):
        runner.score(np.zeros(0, dtype=np.float32))


def test_longer_audio_yields_more_frames(runner):
    short = runner.score(np.zeros(16000, dtype=np.float32))
    long = runner.score(np.zeros(16000 * 5, dtype=np.float32))
    assert long.shape[0] > short.shape[0]

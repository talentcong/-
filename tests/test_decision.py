import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from decision import aggregate_frames, decide, decide_temporal
from event_mapping import EventSpec, LEVEL_ALERT_1, LEVEL_IGNORE, LEVEL_UNMAPPED

N_CLASSES = 521
DOORBELL, MUSIC, UNMAPPED_IDX = 349, 132, 500

LOOKUP = {
    DOORBELL: EventSpec("doorbell", "门铃", 2),
    MUSIC: EventSpec("ignore_132", "音乐", LEVEL_IGNORE),
    UNMAPPED_IDX: EventSpec("__unmapped__", "未覆盖类别", LEVEL_UNMAPPED),
}


def _scores(index_to_score):
    """由 {类别索引: 得分} 构造 521 维得分向量。

    参数是位置传入的 dict，不能用 ** 解包——类别索引是整数，
    而 ** 解包要求键为字符串，会抛 TypeError。
    """
    arr = np.zeros(N_CLASSES, dtype=np.float32)
    for idx, val in index_to_score.items():
        arr[int(idx)] = val
    return arr


def test_aggregate_frames_takes_mean_over_frames():
    frames = np.stack([_scores({DOORBELL: 0.4}), _scores({DOORBELL: 0.8})])
    assert aggregate_frames(frames)[DOORBELL] == np.float32(0.6)


def test_decide_returns_event_above_threshold():
    events = decide(_scores({DOORBELL: 0.7}), LOOKUP, threshold=0.5)
    assert len(events) == 1
    assert events[0].event_id == "doorbell"
    assert events[0].confidence == pytest.approx(0.7)


def test_decide_drops_event_below_threshold():
    assert decide(_scores({DOORBELL: 0.3}), LOOKUP, threshold=0.5) == []


def test_decide_ignores_level3_classes():
    assert decide(_scores({MUSIC: 0.99}), LOOKUP, threshold=0.5) == []


def test_decide_ignores_unmapped_classes():
    assert decide(_scores({UNMAPPED_IDX: 0.99}), LOOKUP, threshold=0.5) == []


def test_decide_merges_same_event_id_keeping_highest_confidence():
    lookup = {
        393: EventSpec("fire_alarm", "火警", LEVEL_ALERT_1),
        394: EventSpec("fire_alarm", "火警", LEVEL_ALERT_1),
    }
    events = decide(_scores({393: 0.6, 394: 0.9}), lookup, threshold=0.5)
    assert len(events) == 1
    assert events[0].confidence == pytest.approx(0.9)


def test_decide_returns_events_sorted_by_confidence_desc():
    events = decide(_scores({DOORBELL: 0.6, 353: 0.9}),
                    {**LOOKUP, 353: EventSpec("knock", "敲门声", 2)},
                    threshold=0.5)
    assert [e.event_id for e in events] == ["knock", "doorbell"]


def test_decide_temporal_requires_min_frames():
    quiet = _scores({DOORBELL: 0.05})
    loud = _scores({DOORBELL: 0.9})
    frames = np.stack([loud, quiet, quiet, quiet, quiet])

    strict = decide_temporal(frames, LOOKUP, threshold=0.5, min_frames=2)
    assert strict == []

    lenient = decide_temporal(frames, LOOKUP, threshold=0.5, min_frames=1)
    assert len(lenient) == 1
    assert lenient[0].event_id == "doorbell"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from detect import build_event_payload, max_level_of


class FakeEvent:
    def __init__(self, event_id, name_cn, level, confidence):
        self.event_id = event_id
        self.name_cn = name_cn
        self.level = level
        self.confidence = confidence

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "name_cn": self.name_cn,
            "level": self.level,
            "level_name": {1: "危险", 2: "注意"}.get(self.level, ""),
            "confidence": self.confidence,
            "audio_set_labels": [],
        }


def test_max_level_of_picks_highest():
    events = [FakeEvent("doorbell", "门铃", 2, 0.7),
              FakeEvent("fire_alarm", "火警", 1, 0.6)]
    assert max_level_of(events) == 1


def test_max_level_of_ignores_level3():
    events = [FakeEvent("ignore_0", "说话声", 3, 0.9)]
    assert max_level_of(events) == 0


def test_max_level_of_empty_is_zero():
    assert max_level_of([]) == 0


def test_build_event_payload_shape():
    events = [FakeEvent("doorbell", "门铃", 2, 0.73)]
    payload = build_event_payload(
        session_id="sample_001",
        audio_file="doorbell_01.wav",
        duration_sec=3.2,
        events=events,
    )
    assert payload["session_id"] == "sample_001"
    assert payload["audio_file"] == "doorbell_01.wav"
    assert payload["duration_sec"] == 3.2
    assert payload["max_level"] == 2
    assert payload["events"][0]["event_id"] == "doorbell"
    assert payload["events"][0]["confidence"] == 0.73


def test_build_event_payload_empty_events():
    payload = build_event_payload("s", "a.wav", 1.0, [])
    assert payload["events"] == []
    assert payload["max_level"] == 0

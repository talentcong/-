import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from event_mapping import (
    LEVEL_ALERT_1,
    LEVEL_ALERT_2,
    LEVEL_IGNORE,
    LEVEL_UNMAPPED,
    build_lookup,
    coverage_report,
)

# 取自实际 yamnet_class_map.csv 的真实索引
SPEECH, DOORBELL, KNOCK, SMOKE, FIRE_ALARM = 0, 349, 353, 393, 394
DOG, BARK, MUSIC, SILENCE = 69, 70, 132, 494
UNLISTED = 500


def _lookup():
    names = {
        SPEECH: "Speech",
        DOORBELL: "Doorbell",
        KNOCK: "Knock",
        SMOKE: "Smoke detector, smoke alarm",
        FIRE_ALARM: "Fire alarm",
        DOG: "Dog",
        BARK: "Bark",
        MUSIC: "Music",
        SILENCE: "Silence",
        UNLISTED: "Zipper",  # 既不在事件表也不在忽略表，且不含忽略关键词
    }
    return build_lookup(names)


def test_doorbell_is_level_2():
    spec = _lookup()[DOORBELL]
    assert spec.event_id == "doorbell"
    assert spec.name_cn == "门铃"
    assert spec.level == LEVEL_ALERT_2


def test_smoke_detector_is_level_1():
    spec = _lookup()[SMOKE]
    assert spec.event_id == "fire_alarm"
    assert spec.level == LEVEL_ALERT_1


def test_smoke_and_fire_alarm_share_event_id():
    lookup = _lookup()
    assert lookup[SMOKE].event_id == lookup[FIRE_ALARM].event_id


def test_dog_and_bark_share_event_id():
    lookup = _lookup()
    assert lookup[DOG].event_id == lookup[BARK].event_id == "dog_bark"


def test_speech_is_ignored():
    spec = _lookup()[SPEECH]
    assert spec.level == LEVEL_IGNORE


def test_music_is_ignored():
    assert _lookup()[MUSIC].level == LEVEL_IGNORE


def test_silence_is_ignored():
    assert _lookup()[SILENCE].level == LEVEL_IGNORE


def test_unlisted_index_is_unmapped():
    assert _lookup()[UNLISTED].level == LEVEL_UNMAPPED


def test_coverage_report_counts_are_consistent():
    lookup = _lookup()
    report = coverage_report(lookup)
    assert report["total"] == len(lookup)
    assert report["alert_1"] > 0
    assert report["alert_2"] > 0
    assert report["ignore"] > 0
    assert report["unmapped"] >= 1
    assert (
        report["alert_1"] + report["alert_2"]
        + report["ignore"] + report["unmapped"] == report["total"]
    )

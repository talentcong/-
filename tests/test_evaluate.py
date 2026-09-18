import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import CATEGORY_TO_EVENT, build_confusion, hit_rate, write_metrics


def test_category_to_event_covers_siren_and_baby():
    assert CATEGORY_TO_EVENT["siren"] == "siren"
    assert CATEGORY_TO_EVENT["crying_baby"] == "baby_cry"
    assert CATEGORY_TO_EVENT["glass_breaking"] == "glass_break"
    assert CATEGORY_TO_EVENT["door_wood_knock"] == "knock"


def test_appliance_categories_expect_silence():
    # 家电声预期不告警，用空字符串表示
    assert CATEGORY_TO_EVENT["vacuum_cleaner"] == ""
    assert CATEGORY_TO_EVENT["washing_machine"] == ""


def test_silence_expectation_hits_when_nothing_predicted():
    assert hit_rate([""], [""]) == 1.0
    assert hit_rate([""], ["siren"]) == 0.0


def test_hit_rate_all_hits():
    assert hit_rate(["siren", "siren"], ["siren", "siren"]) == 1.0


def test_hit_rate_half_hits():
    assert hit_rate(["siren", "siren"], ["siren", "dog_bark"]) == 0.5


def test_hit_rate_empty_is_zero():
    assert hit_rate([], []) == 0.0


def test_build_confusion_counts_pairs():
    expected = ["siren", "siren"]
    predicted = ["siren", "dog_bark"]
    matrix = build_confusion(expected, predicted)
    assert matrix[("siren", "siren")] == 1
    assert matrix[("siren", "dog_bark")] == 1


def test_write_metrics_outputs_csv_json_and_confusion(tmp_path):
    rows = [
        {"filename": "a", "category": "siren",
         "expected_event": "siren", "predicted_event": "siren",
         "confidence": 0.8, "hit": True},
        {"filename": "b", "category": "dog",
         "expected_event": "dog_bark", "predicted_event": "knock",
         "confidence": 0.6, "hit": False},
    ]
    csv_path, json_path, confusion_path = write_metrics(rows, tmp_path)
    assert csv_path.exists() and json_path.exists() and confusion_path.exists()

    text = confusion_path.read_text(encoding="utf-8")
    assert "expected_event,predicted_event,count" in text
    assert "siren,siren,1" in text
    assert "dog_bark,knock,1" in text

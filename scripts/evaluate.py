"""评估：ESC-50 抽样片段 → 听障事件命中情况。

指标口径说明：
ESC-50 的类别与 AudioSet 本体类别体系不同，无法做标准分类准确率。
本脚本统计的是"零样本预测是否落进该类别应有的听障事件族"，
即事件级命中率，不是分类准确率。报告中不得与文献的分类准确率直接比较。

ESC-50 数据许可 CC BY-NC 3.0，仅限教学使用。
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from audio_io import load_audio
from decision import decide, decide_temporal
from event_mapping import build_lookup_from_csv
from yamnet_runner import YamnetRunner

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "yamnet.onnx"
DEFAULT_CLASS_MAP = ROOT / "models" / "yamnet_class_map.csv"
DEFAULT_SAMPLES = ROOT / "data" / "samples"
DEFAULT_OUT = ROOT / "data" / "out"

# ESC-50 类别 → 该类别应对应的听障事件（event_id 取自 event_mapping.EVENT_TABLE）
#
# 空字符串表示"预期不产生任何提示事件"——用于家电类等应当被静默的环境音。
# 这类样本算作命中当且仅当系统确实没有告警，即"不该响的没响"。
CATEGORY_TO_EVENT = {
    "siren": "siren",
    "car_horn": "car_horn",
    "crying_baby": "baby_cry",
    "glass_breaking": "glass_break",
    "clock_alarm": "alarm_clock",
    "door_wood_knock": "knock",
    "door_wood_creaks": "door",
    "dog": "dog_bark",
    "fireworks": "bang",
    "thunderstorm": "thunder",
    "church_bells": "bell",
    "vacuum_cleaner": "",     # 家电运行声，预期静默
    "washing_machine": "",    # 家电运行声，预期静默
}


def hit_rate(expected: list[str], predicted: list[str]) -> float:
    """事件级命中率。expected/predicted 为等长的逐样本事件序列。"""
    if not expected:
        return 0.0
    hits = sum(1 for e, p in zip(expected, predicted) if e == p)
    return hits / len(expected)


def build_confusion(expected: list[str],
                    predicted: list[str]) -> Counter:
    return Counter(zip(expected, predicted))


def _group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def write_metrics(rows: list[dict], out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    fieldnames = ["filename", "category", "expected_event",
                  "predicted_event", "confidence", "hit"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"total": len(rows),
               "hits": sum(1 for r in rows if r["hit"])}
    summary["hit_rate"] = (
        summary["hits"] / summary["total"] if summary["total"] else 0.0
    )
    summary["per_category"] = {
        cat: {
            "n": len(group),
            "hits": sum(1 for r in group if r["hit"]),
            "hit_rate": (sum(1 for r in group if r["hit"]) / len(group)),
        }
        for cat, group in _group_by(rows, "category").items()
    }
    json_path = out_dir / "metrics_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    confusion = build_confusion(
        [r["expected_event"] for r in rows],
        [r["predicted_event"] for r in rows],
    )
    confusion_path = out_dir / "confusion.csv"
    with open(confusion_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["expected_event", "predicted_event", "count"])
        for (expected, predicted), count in sorted(confusion.items()):
            writer.writerow([expected, predicted, count])

    return csv_path, json_path, confusion_path


def load_manifest(samples_dir: Path) -> list[dict]:
    manifest = samples_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"未找到 {manifest}，请先运行: python scripts/download_samples.py"
        )
    with open(manifest, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_sample(runner: YamnetRunner, lookup: dict, class_names: dict,
                    path: Path, threshold: float,
                    min_frames: int) -> tuple[str, float]:
    """返回 (最高等级事件的 event_id, 置信度)。无提示事件时返回 ('', 0.0)。"""
    waveform, _ = load_audio(path)
    frame_scores = runner.score(waveform)
    if min_frames > 1:
        events = decide_temporal(frame_scores, lookup, threshold, min_frames,
                                 class_names)
    else:
        events = decide(frame_scores.mean(axis=0), lookup, threshold,
                        class_names)
    if not events:
        return "", 0.0
    top = events[0]
    return top.event_id, top.confidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ESC-50 抽样评估")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    args = parser.parse_args(argv)

    from download_model import parse_class_map

    class_names = parse_class_map(args.class_map)
    lookup = build_lookup_from_csv(args.class_map)
    runner = YamnetRunner(args.model)

    manifest = load_manifest(args.samples)
    rows: list[dict] = []
    for entry in manifest:
        path = args.samples / f"{entry['filename']}.wav"
        if not path.exists():
            print(f"跳过缺失文件: {path.name}", file=sys.stderr)
            continue
        predicted, confidence = evaluate_sample(
            runner, lookup, class_names, path, args.threshold, args.min_frames
        )
        expected = CATEGORY_TO_EVENT.get(entry["category"], "")
        rows.append({
            "filename": entry["filename"],
            "category": entry["category"],
            "expected_event": expected,
            "predicted_event": predicted,
            "confidence": round(confidence, 4),
            "hit": predicted == expected,
        })
        print(f"{entry['filename']:<14} {entry['category']:<18} "
              f"预期={expected:<12} 预测={predicted or '(无)':<12} "
              f"{'✓' if predicted == expected else '✗'}")

    csv_path, json_path, confusion_path = write_metrics(rows, args.out)
    total = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    rate = hits / total if total else 0.0
    print(f"\n总计 {total} 条，命中 {hits} 条，事件级命中率 {rate:.1%}")
    print(f"明细: {csv_path}")
    print(f"汇总: {json_path}")
    print(f"混淆矩阵: {confusion_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

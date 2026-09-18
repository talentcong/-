"""主流程：音频文件 → 事件 JSON。

用法:
    python scripts/detect.py <音频文件或目录> [--out 输出目录] \
        [--threshold 0.5] [--min-frames 1]
"""
import argparse
import json
import sys
from pathlib import Path

from audio_io import load_audio
from decision import DetectedEvent, aggregate_frames, decide, decide_temporal
from event_mapping import build_lookup_from_csv
from yamnet_runner import YamnetRunner

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASS_MAP = ROOT / "models" / "yamnet_class_map.csv"
DEFAULT_MODEL = ROOT / "models" / "yamnet.onnx"
DEFAULT_OUT = ROOT / "data" / "out"


def max_level_of(events: list) -> int:
    """返回事件列表中最高紧急等级；无可提示事件时返回 0。"""
    levels = [e.level for e in events if e.level in (1, 2)]
    return min(levels) if levels else 0


def build_event_payload(session_id: str, audio_file: str,
                        duration_sec: float,
                        events: list[DetectedEvent]) -> dict:
    """组装写入 JSON、并供人工粘贴到 Coze 的载荷。"""
    return {
        "session_id": session_id,
        "audio_file": audio_file,
        "duration_sec": round(float(duration_sec), 2),
        "max_level": max_level_of(events),
        "events": [e.to_dict() for e in events],
    }


def detect_file(runner: YamnetRunner, lookup: dict, class_names: dict,
                path: Path, threshold: float, min_frames: int) -> dict:
    waveform, sr = load_audio(path)
    frame_scores = runner.score(waveform)

    if min_frames > 1:
        events = decide_temporal(frame_scores, lookup, threshold, min_frames,
                                 class_names)
    else:
        events = decide(aggregate_frames(frame_scores), lookup, threshold,
                        class_names)

    return build_event_payload(
        session_id=path.stem,
        audio_file=path.name,
        duration_sec=len(waveform) / sr,
        events=events,
    )


def collect_inputs(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(
            p for p in target.iterdir()
            if p.suffix.lower() in (".wav", ".flac", ".ogg")
        )
    return [target]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="环境声音事件检测")
    parser.add_argument("target", type=Path, help="音频文件或目录")
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

    files = collect_inputs(args.target)
    if not files:
        print(f"未找到音频文件: {args.target}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for path in files:
        payload = detect_file(runner, lookup, class_names, path,
                              args.threshold, args.min_frames)
        out_path = args.out / f"{path.stem}.json"
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = "、".join(
            f"{e['name_cn']}({e['confidence']:.2f})" for e in payload["events"]
        ) or "无提示事件"
        print(f"{path.name}: {summary}")

    print(f"\nJSON 输出目录: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

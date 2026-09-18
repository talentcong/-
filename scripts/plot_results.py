"""从指标汇总 JSON 生成报告用图表。

只负责读汇总结果与画图，不做指标计算——指标计算属于 evaluate.py。
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt  # noqa: E402

# matplotlib 默认字体 DejaVu Sans 不含 CJK，中文标签会渲染成豆腐块。
# 必须在画任何图之前设置。
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "out"


def load_sweep(out_dir: Path) -> list[dict]:
    """读取 thr_* 与 mf_* 目录下的 metrics_summary.json。

    返回按类别排序的列表，每项含 label / hit_rate / kind / threshold|min_frames。
    """
    rows: list[dict] = []

    for path in sorted(out_dir.glob("thr_*/metrics_summary.json")):
        name = path.parent.name
        try:
            threshold = float(name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "kind": "threshold",
            "threshold": threshold,
            "label": f"threshold={threshold:.2f}",
            "hit_rate": summary["hit_rate"],
            "hits": summary["hits"],
            "total": summary["total"],
        })

    for path in sorted(out_dir.glob("mf_*/metrics_summary.json")):
        name = path.parent.name
        try:
            min_frames = int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "kind": "min_frames",
            "min_frames": min_frames,
            "label": f"min_frames={min_frames}",
            "hit_rate": summary["hit_rate"],
            "hits": summary["hits"],
            "total": summary["total"],
        })

    threshold_rows = sorted(
        (r for r in rows if r["kind"] == "threshold"),
        key=lambda r: r["threshold"],
    )
    frame_rows = sorted(
        (r for r in rows if r["kind"] == "min_frames"),
        key=lambda r: r["min_frames"],
    )
    return threshold_rows + frame_rows


def plot_sweep(rows: list[dict], out_path: Path) -> Path:
    """画阈值扫描曲线与去抖窗口对比柱状图。"""
    threshold_rows = [r for r in rows if r["kind"] == "threshold"]
    frame_rows = [r for r in rows if r["kind"] == "min_frames"]
    if not threshold_rows and not frame_rows:
        raise ValueError("没有可绘制的数据，请先运行 evaluate.py 的参数扫描")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if threshold_rows:
        axes[0].plot(
            [r["threshold"] for r in threshold_rows],
            [r["hit_rate"] for r in threshold_rows],
            marker="o",
        )
        axes[0].set_xlabel("阈值 threshold")
        axes[0].set_ylabel("事件级命中率")
        axes[0].set_title("阈值对命中率的影响")
        axes[0].grid(alpha=0.3)

    if frame_rows:
        axes[1].bar(
            [str(r["min_frames"]) for r in frame_rows],
            [r["hit_rate"] for r in frame_rows],
        )
        axes[1].set_xlabel("最少命中帧数 min_frames")
        axes[1].set_ylabel("事件级命中率")
        axes[1].set_title("去抖窗口对命中率的影响")
        axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成报告图表")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure", type=Path,
                        default=DEFAULT_OUT / "threshold_sweep.png")
    args = parser.parse_args(argv)

    rows = load_sweep(args.out)
    if not rows:
        print("未找到扫描结果，请先运行 evaluate.py 的参数扫描", file=sys.stderr)
        return 1
    path = plot_sweep(rows, args.figure)
    print(f"图表已生成: {path}")
    for row in rows:
        print(f"  {row['label']:<18} 命中率 {row['hit_rate']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_results import load_sweep


def test_load_sweep_reads_threshold_and_minframes_dirs(tmp_path):
    for name, rate in (("thr_0.10", 0.4), ("thr_0.50", 0.1)):
        d = tmp_path / name
        d.mkdir()
        (d / "metrics_summary.json").write_text(
            json.dumps({"total": 10, "hits": int(rate * 10), "hit_rate": rate}),
            encoding="utf-8",
        )
    for name, rate in (("mf_1", 0.4), ("mf_3", 0.2)):
        d = tmp_path / name
        d.mkdir()
        (d / "metrics_summary.json").write_text(
            json.dumps({"total": 10, "hits": int(rate * 10), "hit_rate": rate}),
            encoding="utf-8",
        )

    rows = load_sweep(tmp_path)
    labels = [r["label"] for r in rows]
    assert "threshold=0.10" in labels
    assert "threshold=0.50" in labels
    assert "min_frames=1" in labels
    assert "min_frames=3" in labels


def test_load_sweep_sorts_thresholds_numerically(tmp_path):
    for name in ("thr_0.50", "thr_0.05", "thr_0.20"):
        d = tmp_path / name
        d.mkdir()
        (d / "metrics_summary.json").write_text(
            json.dumps({"total": 1, "hits": 1, "hit_rate": 1.0}), encoding="utf-8"
        )
    rows = load_sweep(tmp_path)
    thresholds = [r["threshold"] for r in rows if r["kind"] == "threshold"]
    assert thresholds == sorted(thresholds)


def test_load_sweep_skips_missing_dirs(tmp_path):
    assert load_sweep(tmp_path) == []

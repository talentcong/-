import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_results import load_sweep, plot_sweep


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


def test_cjk_font_is_configured():
    import matplotlib
    import plot_results  # noqa: F401  触发模块级 rcParams 设置

    font_list = matplotlib.rcParams["font.sans-serif"]
    assert "Microsoft YaHei" in font_list or "SimHei" in font_list
    assert matplotlib.rcParams["axes.unicode_minus"] is False


def test_plot_sweep_renders_chinese_without_missing_glyphs(tmp_path):
    """中文字形缺失时 matplotlib 会发 UserWarning，这里把它升级为错误。

    若字体配置被误删，本测试会失败——这是对上面那条修复的回归保护。
    """
    import warnings

    rows = [{
        "kind": "threshold", "threshold": 0.10, "label": "threshold=0.10",
        "hit_rate": 0.4, "hits": 4, "total": 10,
    }]
    out = tmp_path / "fig.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_sweep(rows, out)
    assert out.exists() and out.stat().st_size > 0

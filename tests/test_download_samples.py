import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_samples import (
    BASE_AUDIO_URL,
    HEARING_RELEVANT_CATEGORIES,
    _force_ipv4,
    parse_metadata,
    pick_filenames,
)


def test_parse_metadata_strips_wav_extension():
    """ESC-50 的 filename 列自带 .wav 后缀，必须去掉，否则拼接会重复。"""
    text = (
        "filename,fold,target,category,esc10,src_file,take\n"
        "1-100032-A-0.wav,1,0,dog,True,100032,A\n"
        "5-210612-A-37.wav,5,37,siren,False,210612,A\n"
    )
    rows = parse_metadata(text)
    assert rows[0]["filename"] == "1-100032-A-0"
    assert rows[1]["filename"] == "5-210612-A-37"
    assert rows[1]["category"] == "siren"


def test_force_ipv4_restricts_address_family():
    """在 _force_ipv4 上下文内，DNS 解析只能返回 IPv4。"""
    with _force_ipv4():
        infos = socket.getaddrinfo("localhost", 80)
    assert infos
    assert all(f[0] == socket.AF_INET for f in infos)


def test_force_ipv4_restores_getaddrinfo():
    """退出上下文后必须还原，不能污染全局状态。"""
    original = socket.getaddrinfo
    with _force_ipv4():
        assert socket.getaddrinfo is not original
    assert socket.getaddrinfo is original


def test_base_url_is_github_raw():
    assert BASE_AUDIO_URL.startswith("https://raw.githubusercontent.com/")
    assert "karolpiczak/ESC-50" in BASE_AUDIO_URL


def test_hearing_relevant_categories_are_nonempty():
    assert len(HEARING_RELEVANT_CATEGORIES) >= 10
    assert "siren" in HEARING_RELEVANT_CATEGORIES
    assert "crying_baby" in HEARING_RELEVANT_CATEGORIES


def test_pick_filenames_selects_n_per_category():
    rows = [
        {"filename": f"a{i}", "category": "siren", "fold": "1"} for i in range(40)
    ] + [
        {"filename": f"b{i}", "category": "dog", "fold": "1"} for i in range(40)
    ]
    picked = pick_filenames(rows, ["siren", "dog"], per_category=5)
    assert len(picked) == 10
    assert sum(1 for r in picked if r["category"] == "siren") == 5
    assert sum(1 for r in picked if r["category"] == "dog") == 5


def test_pick_filenames_is_deterministic_and_sorted():
    rows = [
        {"filename": f"a{i}", "category": "siren", "fold": "1"} for i in range(20)
    ]
    first = pick_filenames(rows, ["siren"], per_category=3)
    second = pick_filenames(rows, ["siren"], per_category=3)
    assert first == second
    assert first == sorted(first, key=lambda r: r["filename"])


def test_pick_filenames_spreads_across_folds():
    rows = [
        {"filename": f"f{fold}-{i}", "category": "siren", "fold": str(fold)}
        for fold in range(1, 11) for i in range(4)
    ]
    picked = pick_filenames(rows, ["siren"], per_category=5)
    assert len({r["fold"] for r in picked}) == 5

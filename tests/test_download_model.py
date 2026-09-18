import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_model import (
    CLASS_MAP_URL,
    MODEL_URL,
    download,
    parse_class_map,
)


def test_urls_use_hf_mirror():
    assert "hf-mirror.com" in MODEL_URL
    assert "hf-mirror.com" in CLASS_MAP_URL
    assert "huggingface.co" not in MODEL_URL


def test_parse_class_map_reads_index_and_display_name(tmp_path):
    csv_path = tmp_path / "cm.csv"
    csv_path.write_text(
        'index,mid,display_name\n'
        '0,/m/09x0r,Speech\n'
        '349,/m/0gywn,Doorbell\n'
        '393,/m/0d3z1,"Smoke detector, smoke alarm"\n',
        encoding="utf-8",
    )
    result = parse_class_map(csv_path)
    assert len(result) == 3
    assert result[0] == "Speech"
    assert result[349] == "Doorbell"
    assert result[393] == "Smoke detector, smoke alarm"


class _FakeResponse:
    """模拟 urllib 响应对象，逐块吐出预置数据。"""

    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self, size=-1):
        chunk, self._data = self._data[:size], self._data[size:]
        return chunk


def _patch_urlopen(monkeypatch, data: bytes) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: _FakeResponse(data)
    )


def test_download_writes_dest_and_leaves_no_part(tmp_path, monkeypatch):
    payload = b"abc" * 100
    _patch_urlopen(monkeypatch, payload)
    dest = tmp_path / "m.onnx"
    download("http://example.com/m.onnx", dest, expected_size=len(payload))
    assert dest.read_bytes() == payload
    assert [p.name for p in tmp_path.iterdir()] == ["m.onnx"]


def test_download_rejects_truncated_body(tmp_path, monkeypatch):
    """响应被截断时必须报错并丢弃残file，而不是静默落盘。"""
    _patch_urlopen(monkeypatch, b"x" * 1000)
    dest = tmp_path / "m.onnx"
    with pytest.raises(RuntimeError, match="下载不完整"):
        download("http://example.com/m.onnx", dest, expected_size=16124200)
    assert list(tmp_path.iterdir()) == []


def test_download_cleans_part_when_read_raises(tmp_path, monkeypatch):
    class _Boom:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self, size=-1):
            raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Boom())
    dest = tmp_path / "m.onnx"
    with pytest.raises(RuntimeError, match="下载失败"):
        download("http://example.com/m.onnx", dest, expected_size=10)
    assert list(tmp_path.iterdir()) == []


def test_download_skips_existing_nonempty_file(tmp_path, monkeypatch):
    def _unexpected(*args, **kwargs):
        raise AssertionError("已存在的文件不应触发网络请求")

    monkeypatch.setattr(urllib.request, "urlopen", _unexpected)
    dest = tmp_path / "m.onnx"
    dest.write_bytes(b"already here")
    download("http://example.com/m.onnx", dest, expected_size=999)
    assert dest.read_bytes() == b"already here"

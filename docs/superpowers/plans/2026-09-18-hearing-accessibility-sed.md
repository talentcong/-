# 听障无障碍环境声音提示智能体 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个听障无障碍环境声音提示系统：本地用 YAMNet ONNX 做零样本声学事件检测，输出分级事件 JSON，再由 Coze 智能体生成中文提示文案，最终产出一份实验报告。

**Architecture:** 两条链路。链路 A（Python）：音频 → 预处理 → YAMNet ONNX 推理 → 521 类 AudioSet 得分 → 映射层分级归并 → 事件 JSON。链路 B（Coze）：人工把 JSON 粘贴进单节点工作流，大模型节点生成提示文案。两条链路通过 JSON 契约衔接。

**Tech Stack:** Python 3.13、onnxruntime 1.30.0、numpy、scipy、pandas、matplotlib、pytest；模型为 YAMNet ONNX（16.1 MB，经 hf-mirror 下载）；评估数据为 ESC-50 抽样片段（经 raw.githubusercontent.com 下载）。

---

## 前置核实结果（已在规划阶段实测，非假设）

| 事项 | 结论 |
|---|---|
| `onnxruntime` 在 Python 3.13 | **可用**。PyPI 上 `onnxruntime-1.30.0-cp313-cp313-win_amd64.whl` 存在，`requires_python >=3.11` |
| `huggingface.co` | 不通（HTTP 000），**必须走 `hf-mirror.com`** |
| 模型文件 | `hf-mirror.com/niobures/YAMNet/resolve/main/yamnetonnx/yamnet.onnx`，16,124,200 字节，仓库无门禁 |
| 类别表 | 同路径 `yamnet_class_map.csv`，14,096 字节，521 条，列为 `index,mid,display_name` |
| ESC-50 单条音频 | `raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{filename}.wav` 可达，每条 441,044 字节（16-bit 单声道 44.1 kHz，5 秒） |
| ESC-50 标注 | `raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv` 可达，93,742 字节 |
| ESC-50 许可证 | CC BY-NC 3.0（署名—非商业），须在报告第六章声明 |

## 与设计文档的一处偏离

设计文档 4.3 节把"映射表"与"去抖决策"都放在 `event_mapping.py`。实施时**拆成两个文件**：

- `event_mapping.py` —— 静态知识：映射表、分级、覆盖率统计
- `decision.py` —— 决策逻辑：帧聚合、阈值、多帧去抖

理由：前者是数据，后者是算法，变更原因完全不同，混在一起会让文件同时承担两种职责。这是设计文档模块表的细化，不改变任何对外行为。

## 文件结构

| 文件 | 职责 |
|---|---|
| `requirements.txt` | 依赖清单 |
| `scripts/download_model.py` | 经 hf-mirror 下载模型与类别表；解析类别表 |
| `scripts/audio_io.py` | 音频读取、重采样到 16 kHz、转单声道、峰值归一化 |
| `scripts/event_mapping.py` | 事件映射表、紧急分级、索引查找、覆盖率统计 |
| `scripts/decision.py` | 帧级得分的聚合、阈值判定、多帧一致性去抖 |
| `scripts/yamnet_runner.py` | ONNX 会话封装与推理 |
| `scripts/make_test_signals.py` | 生成合成测试信号 |
| `scripts/detect.py` | 主流程：音频 → 事件 JSON |
| `scripts/download_samples.py` | ESC-50 抽样片段下载 |
| `scripts/evaluate.py` | 指标统计与混淆矩阵 |
| `scripts/plot_results.py` | 从指标汇总 JSON 生成报告用图表 |
| `tests/test_*.py` | 各模块单元测试 |
| `data/out/` | 事件 JSON、指标表、图表 |
| `docs/听障无障碍环境声音提示智能体实验报告.docx` | 最终交付物 |

---

## Task 1: 项目骨架与依赖安装

**Files:**
- Create: `requirements.txt`
- Modify: `.gitignore`（已存在，无需改动）

- [ ] **Step 1: 写 `requirements.txt`**

```
# 推理引擎与数值计算：精确锁定。
# 报告第五章的得分分布、命中率、阈值扫描数据都由这几个版本产生，
# 浮动版本会让实验结果无法复现。
onnxruntime==1.30.0
numpy==2.4.6
scipy==1.17.1

# 绘图与文档生成：不参与数值计算，用下限约束即可
matplotlib>=3.8
python-docx>=1.1

# 测试
pytest>=8.0
```

- [ ] **Step 2: 安装依赖**

Run:
```bash
cd "c:/Users/ASUS/Desktop/数字音频处理/音频智能体实验"
python -m pip install -r requirements.txt
```
Expected: 成功安装 `onnxruntime-1.30.0`，无 "No matching distribution found" 报错。

- [ ] **Step 3: 验证关键依赖可导入**

Run:
```bash
python -c "import onnxruntime, numpy, scipy, matplotlib, pytest, docx; print('onnxruntime', onnxruntime.__version__); print('all imports OK')"
```
Expected:
```
onnxruntime 1.30.0
all imports OK
```

- [ ] **Step 4: 建目录**

Run:
```bash
mkdir -p scripts tests models data/samples data/signals data/out
```

- [ ] **Step 5: 提交**

```bash
git add requirements.txt
git commit -m "添加项目依赖清单"
```

---

## Task 2: 模型与类别表下载

**Files:**
- Create: `scripts/download_model.py`
- Create: `tests/test_download_model.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_download_model.py`:
```python
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
    """响应被截断时必须报错并丢弃残缺文件，而不是静默落盘。"""
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


def test_download_skips_existing_complete_file(tmp_path, monkeypatch):
    payload = b"already here"

    def _unexpected(*args, **kwargs):
        raise AssertionError("完整文件不应触发网络请求")

    monkeypatch.setattr(urllib.request, "urlopen", _unexpected)
    dest = tmp_path / "m.onnx"
    dest.write_bytes(payload)
    download("http://example.com/m.onnx", dest, expected_size=len(payload))
    assert dest.read_bytes() == payload


def test_download_redownloads_when_existing_size_is_wrong(tmp_path, monkeypatch):
    """非空但残缺的既有文件必须被重新下载，而不是永久跳过。"""
    payload = b"z" * 500
    _patch_urlopen(monkeypatch, payload)
    dest = tmp_path / "m.onnx"
    dest.write_bytes(b"truncated")
    download("http://example.com/m.onnx", dest, expected_size=len(payload))
    assert dest.read_bytes() == payload
    assert [p.name for p in tmp_path.iterdir()] == ["m.onnx"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_download_model.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'download_model'`

- [ ] **Step 3: 写实现**

`scripts/download_model.py`:
```python
"""下载 YAMNet ONNX 模型与 AudioSet 类别表。

huggingface.co 在本机直连不通，必须走 hf-mirror.com 镜像。
"""
import csv
import sys
import urllib.request
from pathlib import Path

MIRROR = "https://hf-mirror.com"
REPO = "niobures/YAMNet"
REVISION = "main"

MODEL_URL = f"{MIRROR}/{REPO}/resolve/{REVISION}/yamnetonnx/yamnet.onnx"
CLASS_MAP_URL = f"{MIRROR}/{REPO}/resolve/{REVISION}/yamnetonnx/yamnet_class_map.csv"

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "yamnet.onnx"
CLASS_MAP_PATH = MODELS_DIR / "yamnet_class_map.csv"

EXPECTED_CLASS_COUNT = 521
MODEL_SIZE = 16_124_200
CLASS_MAP_SIZE = 14_096


def download(url: str, dest: Path, expected_size: int) -> None:
    """下载 url 到 dest。dest 已存在且大小正确则跳过。

    expected_size 同时承担两个作用：
    1) 校验新下载的完整性——HTTP 响应被提前截断时 http.client 不会抛异常
       （显式传 amt 给 read() 时它只返回空串），不校验就会把残缺文件落盘；
    2) 判断本地已有文件是否可信——只看"非空"识别不出残缺文件，
       会让坏文件被永久跳过，并在后续任务中报出无关的错误。
    """
    if dest.exists():
        existing = dest.stat().st_size
        if existing == expected_size:
            print(f"已存在，跳过: {dest.name}")
            return
        print(f"已存在但大小不符（{existing} 字节，期望 {expected_size}），重新下载: {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"下载 {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
            while chunk := resp.read(1 << 16):
                f.write(chunk)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"下载失败: {url}\n原因: {exc}\n"
            f"请检查网络；若 hf-mirror.com 不可用，可手动下载后放到 {dest}"
        ) from exc

    actual_size = tmp.stat().st_size
    if actual_size != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"下载不完整: {url}\n"
            f"期望 {expected_size} 字节，实际 {actual_size} 字节\n"
            f"文件已丢弃，请重新运行本脚本"
        )
    tmp.replace(dest)


def parse_class_map(path: Path) -> dict[int, str]:
    """解析 yamnet_class_map.csv，返回 {类别索引: 显示名}。"""
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["index"]): row["display_name"] for row in csv.DictReader(f)}


def main() -> int:
    download(MODEL_URL, MODEL_PATH, MODEL_SIZE)
    download(CLASS_MAP_URL, CLASS_MAP_PATH, CLASS_MAP_SIZE)

    mapping = parse_class_map(CLASS_MAP_PATH)
    if len(mapping) != EXPECTED_CLASS_COUNT:
        print(
            f"错误: 类别表应有 {EXPECTED_CLASS_COUNT} 类，实际 {len(mapping)} 类",
            file=sys.stderr,
        )
        return 1

    print(f"模型: {MODEL_PATH} ({MODEL_PATH.stat().st_size} 字节)")
    print(f"类别表: {CLASS_MAP_PATH} ({len(mapping)} 类)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_download_model.py -v`
Expected: 7 passed

- [ ] **Step 5: 实际下载并校验**

Run: `python scripts/download_model.py`
Expected:
```
模型: ...\models\yamnet.onnx (16124200 字节)
类别表: ...\models\yamnet_class_map.csv (521 类)
```

- [ ] **Step 6: 提交**

```bash
git add scripts/download_model.py tests/test_download_model.py
git commit -m "添加 YAMNet 模型与类别表下载脚本"
```

---

## Task 3: 音频预处理

**Files:**
- Create: `scripts/audio_io.py`
- Create: `tests/test_audio_io.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_audio_io.py`:
```python
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audio_io import TARGET_SR, load_audio, normalize_peak, to_mono


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    assert np.allclose(to_mono(stereo), [2.0, 3.0])


def test_to_mono_passes_through_mono():
    mono = np.array([1.0, 2.0], dtype=np.float32)
    assert np.allclose(to_mono(mono), mono)


def test_normalize_peak_scales_to_unit_range():
    assert np.allclose(normalize_peak(np.array([2.0, -4.0], dtype=np.float32)),
                       [0.5, -1.0])


def test_normalize_peak_leaves_silence_untouched():
    assert np.allclose(normalize_peak(np.zeros(4, dtype=np.float32)), np.zeros(4))


def test_load_audio_resamples_44100_to_16000(tmp_path):
    sr_in = 44100
    t = np.arange(sr_in, dtype=np.float32) / sr_in
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    path = tmp_path / "tone44k.wav"
    wavfile.write(path, sr_in, tone)

    audio, sr = load_audio(path)
    assert sr == TARGET_SR
    assert audio.dtype == np.float32
    assert len(audio) == pytest.approx(TARGET_SR, abs=2)
    assert np.max(np.abs(audio)) == pytest.approx(1.0, abs=1e-3)


def test_load_audio_writes_nothing_and_handles_stereo(tmp_path):
    sr_in = 44100
    left = np.full(sr_in, 1000, dtype=np.int16)
    right = np.full(sr_in, -3000, dtype=np.int16)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "stereo.wav"
    wavfile.write(path, sr_in, stereo)

    audio, sr = load_audio(path)
    assert sr == TARGET_SR
    assert audio.ndim == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_audio_io.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'audio_io'`

- [ ] **Step 3: 写实现**

`scripts/audio_io.py`:
```python
"""音频读取与预处理：转单声道、重采样到 16 kHz、峰值归一化。

YAMNet 要求 16 kHz 单声道、幅值落在 [-1, 1] 的 float32 波形。
"""
from math import gcd
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly

TARGET_SR = 16000


def to_mono(audio: np.ndarray) -> np.ndarray:
    """多声道取各声道均值，单声道原样返回。"""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    """峰值归一化到 [-1, 1]。全零（静音）输入原样返回，避免除零。"""
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    return audio / peak


def to_float32(audio: np.ndarray) -> np.ndarray:
    """整型 PCM 转为 [-1, 1] 的 float32；已是浮点则直接转类型。"""
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        return (audio.astype(np.float32) / scale).astype(np.float32)
    return audio.astype(np.float32)


def resample_to_target(audio: np.ndarray, sr_in: int,
                       sr_out: int = TARGET_SR) -> np.ndarray:
    """重采样到目标采样率。采样率已匹配则原样返回。"""
    if sr_in == sr_out:
        return audio
    divisor = gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // divisor
    down = int(sr_in) // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def load_audio(path: str | Path, sr_out: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """读取音频文件，返回 (单声道 float32 波形, 采样率)。

    依次执行：整型转浮点 → 转单声道 → 重采样 → 峰值归一化。
    """
    sr_in, raw = wavfile.read(str(path))
    audio = to_float32(raw)
    audio = to_mono(audio)
    audio = resample_to_target(audio, sr_in, sr_out)
    audio = normalize_peak(audio)
    return audio.astype(np.float32), sr_out
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_audio_io.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add scripts/audio_io.py tests/test_audio_io.py
git commit -m "添加音频读取与预处理模块"
```

---

## Task 4: 事件映射表与紧急分级

映射表中的类别索引**全部来自实际下载的 `yamnet_class_map.csv`，已逐条核对**，不是凭记忆填写。

**Files:**
- Create: `scripts/event_mapping.py`
- Create: `tests/test_event_mapping.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_event_mapping.py`:
```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_event_mapping.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'event_mapping'`

- [ ] **Step 3: 写实现**

`scripts/event_mapping.py`:
```python
"""AudioSet 521 类 → 听障关注事件的映射与紧急分级。

类别索引取自实际下载的 yamnet_class_map.csv，逐条核对过。
同一事件族共享 event_id，避免同一声源重复告警。

等级约定：
  1 = 危险，立即提示
  2 = 注意，需要知晓
  3 = 已识别但可忽略的环境音
  0 = 未覆盖，AudioSet 本体不含该声音（覆盖盲区，不是"可忽略"）
"""
from dataclasses import dataclass
from pathlib import Path

LEVEL_ALERT_1 = 1
LEVEL_ALERT_2 = 2
LEVEL_IGNORE = 3
LEVEL_UNMAPPED = 0

LEVEL_NAMES = {
    LEVEL_ALERT_1: "危险",
    LEVEL_ALERT_2: "注意",
    LEVEL_IGNORE: "可忽略",
    LEVEL_UNMAPPED: "未覆盖",
}


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    name_cn: str
    level: int


# 事件族定义：event_id → (中文名, 等级, [AudioSet 类别索引])
EVENT_TABLE: dict[str, tuple[str, int, list[int]]] = {
    # ---- 一级：危险，立即提示 ----
    "fire_alarm":   ("火警",        LEVEL_ALERT_1, [393, 394]),
    "siren":        ("警报声",      LEVEL_ALERT_1, [316, 317, 318, 319, 390, 391]),
    "car_alarm":    ("汽车防盗警报", LEVEL_ALERT_1, [304]),
    "explosion":    ("爆炸声",      LEVEL_ALERT_1, [420, 424]),
    "gunshot":      ("枪声",        LEVEL_ALERT_1, [421, 422, 425]),
    "glass_break":  ("玻璃破碎",    LEVEL_ALERT_1, [435, 437]),
    "screaming":    ("尖叫声",      LEVEL_ALERT_1, [11]),
    "alarm":        ("警报器",      LEVEL_ALERT_1, [382]),
    "fire":         ("燃烧声",      LEVEL_ALERT_1, [292]),
    "bang":         ("爆响",        LEVEL_ALERT_1, [460, 426, 427, 428]),
    # ---- 二级：注意，需要知晓 ----
    "doorbell":     ("门铃",        LEVEL_ALERT_2, [349]),
    "knock":        ("敲门声",      LEVEL_ALERT_2, [353]),
    "door":         ("开关门声",    LEVEL_ALERT_2, [348, 351]),
    "phone_ring":   ("电话铃声",    LEVEL_ALERT_2, [383, 384, 385]),
    "baby_cry":     ("婴儿哭声",    LEVEL_ALERT_2, [19, 20]),
    "dog_bark":     ("狗叫声",      LEVEL_ALERT_2, [69, 70, 75, 117]),
    "car_horn":     ("汽车鸣笛",    LEVEL_ALERT_2, [302, 312]),
    "alarm_clock":  ("闹钟",        LEVEL_ALERT_2, [389]),
    "buzzer":       ("蜂鸣器",      LEVEL_ALERT_2, [392]),
    "whistle":      ("哨声/汽笛",   LEVEL_ALERT_2, [395, 396, 397, 324, 325]),
    "bell":         ("铃声",        LEVEL_ALERT_2, [195, 196, 198, 202]),
    "beep":         ("电子提示音",  LEVEL_ALERT_2, [475, 313]),
    "thunder":      ("雷声",        LEVEL_ALERT_2, [280, 281]),
}

# 可忽略环境音的判定关键词，作用于 display_name（小写匹配）
IGNORE_KEYWORDS = [
    "music", "speech", "silence", "noise", "static", "hum",
    "traffic", "vehicle", "engine", "car", "bus", "train", "subway",
    "aircraft", "airplane", "helicopter", "motorcycle", "boat",
    "vacuum", "air conditioning", "fan",
    "typing", "footstep", "walk",
]


def _ignored(event_id: str, name_cn: str) -> EventSpec:
    return EventSpec(event_id, name_cn, LEVEL_IGNORE)


# 可忽略类别：索引 → 中文名。索引取自实际类别表。
IGNORE_TABLE: dict[int, str] = {
    0: "说话声", 1: "儿童说话声", 5: "语音合成器", 12: "耳语", 65: "嘈杂人声",
    132: "音乐", 133: "乐器", 211: "流行乐", 212: "嘻哈乐", 214: "摇滚乐",
    232: "古典乐", 262: "背景音乐", 265: "影视配乐", 269: "舞曲",
    48: "脚步声", 378: "打字声",
    294: "车辆声", 300: "机动车声", 301: "汽车声", 308: "车辆驶过",
    315: "公交车声", 320: "摩托车声", 321: "交通噪声", 323: "火车声",
    328: "地铁声", 330: "飞机引擎声", 331: "喷气引擎声", 333: "直升机声",
    334: "固定翼飞机声", 337: "引擎声", 338: "高频引擎声",
    342: "中频引擎声", 343: "低频引擎声", 345: "引擎启动声", 346: "怠速声",
    371: "吸尘器声", 406: "机械风扇声", 407: "空调声",
    494: "静音", 507: "噪声", 508: "环境噪声", 509: "静电噪声",
    510: "电源交流声", 514: "白噪声", 515: "粉红噪声", 279: "风噪",
}

UNMAPPED_SPEC = EventSpec("__unmapped__", "未覆盖类别", LEVEL_UNMAPPED)


def build_lookup(class_names: dict[int, str]) -> dict[int, EventSpec]:
    """构建 类别索引 → EventSpec 的查找表。

    class_names 为 {索引: display_name}，通常来自 download_model.parse_class_map。
    优先级：事件表 > 忽略表 > 关键词匹配 > 未覆盖。
    """
    lookup: dict[int, EventSpec] = {}

    for event_id, (name_cn, level, indices) in EVENT_TABLE.items():
        for index in indices:
            lookup[index] = EventSpec(event_id, name_cn, level)

    for index, name_cn in IGNORE_TABLE.items():
        lookup.setdefault(index, _ignored(f"ignore_{index}", name_cn))

    for index, display_name in class_names.items():
        if index in lookup:
            continue
        lowered = display_name.lower()
        if any(keyword in lowered for keyword in IGNORE_KEYWORDS):
            lookup[index] = _ignored(f"ignore_{index}", "可忽略环境音")
        else:
            lookup[index] = UNMAPPED_SPEC

    return lookup


def build_lookup_from_csv(csv_path: str | Path) -> dict[int, EventSpec]:
    """从 yamnet_class_map.csv 直接构建查找表。"""
    from download_model import parse_class_map

    return build_lookup(parse_class_map(Path(csv_path)))


def coverage_report(lookup: dict[int, EventSpec]) -> dict[str, int]:
    """统计映射覆盖率：各级别类别数与未覆盖数。"""
    counts = {
        "total": len(lookup),
        "alert_1": 0,
        "alert_2": 0,
        "ignore": 0,
        "unmapped": 0,
    }
    for spec in lookup.values():
        if spec.level == LEVEL_ALERT_1:
            counts["alert_1"] += 1
        elif spec.level == LEVEL_ALERT_2:
            counts["alert_2"] += 1
        elif spec.level == LEVEL_IGNORE:
            counts["ignore"] += 1
        else:
            counts["unmapped"] += 1
    return counts
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_event_mapping.py -v`
Expected: 9 passed

- [ ] **Step 5: 用真实类别表验证覆盖统计**

Run:
```bash
python -c "
import sys; sys.path.insert(0,'scripts')
from event_mapping import build_lookup_from_csv, coverage_report
lookup = build_lookup_from_csv('models/yamnet_class_map.csv')
r = coverage_report(lookup)
print(r)
assert r['total'] == 521, r
assert r['alert_1'] > 0 and r['alert_2'] > 0
print('覆盖统计校验通过')
"
```
Expected: 打印统计字典，`total` 为 521，末行 `覆盖统计校验通过`。**把输出记录下来，这是报告第五章的数据。**

- [ ] **Step 6: 提交**

```bash
git add scripts/event_mapping.py tests/test_event_mapping.py
git commit -m "添加 AudioSet 到听障事件的映射表与紧急分级"
```

---

## Task 5: 决策逻辑（聚合、阈值、去抖）

**Files:**
- Create: `scripts/decision.py`
- Create: `tests/test_decision.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_decision.py`:
```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from decision import aggregate_frames, decide, decide_temporal
from event_mapping import EventSpec, LEVEL_ALERT_1, LEVEL_IGNORE, LEVEL_UNMAPPED

N_CLASSES = 521
DOORBELL, MUSIC, UNMAPPED_IDX = 349, 132, 500

LOOKUP = {
    DOORBELL: EventSpec("doorbell", "门铃", 2),
    MUSIC: EventSpec("ignore_132", "音乐", LEVEL_IGNORE),
    UNMAPPED_IDX: EventSpec("__unmapped__", "未覆盖类别", LEVEL_UNMAPPED),
}


def _scores(index_to_score):
    """由 {类别索引: 得分} 构造 521 维得分向量。

    参数是位置传入的 dict，不能用 ** 解包——类别索引是整数，
    而 ** 解包要求键为字符串，会抛 TypeError。
    """
    arr = np.zeros(N_CLASSES, dtype=np.float32)
    for idx, val in index_to_score.items():
        arr[int(idx)] = val
    return arr


def test_aggregate_frames_takes_mean_over_frames():
    frames = np.stack([_scores({DOORBELL: 0.4}), _scores({DOORBELL: 0.8})])
    assert aggregate_frames(frames)[DOORBELL] == np.float32(0.6)


def test_decide_returns_event_above_threshold():
    events = decide(_scores({DOORBELL: 0.7}), LOOKUP, threshold=0.5)
    assert len(events) == 1
    assert events[0].event_id == "doorbell"
    assert events[0].confidence == pytest.approx(0.7)


def test_decide_drops_event_below_threshold():
    assert decide(_scores({DOORBELL: 0.3}), LOOKUP, threshold=0.5) == []


def test_decide_ignores_level3_classes():
    assert decide(_scores({MUSIC: 0.99}), LOOKUP, threshold=0.5) == []


def test_decide_ignores_unmapped_classes():
    assert decide(_scores({UNMAPPED_IDX: 0.99}), LOOKUP, threshold=0.5) == []


def test_decide_merges_same_event_id_keeping_highest_confidence():
    lookup = {
        393: EventSpec("fire_alarm", "火警", LEVEL_ALERT_1),
        394: EventSpec("fire_alarm", "火警", LEVEL_ALERT_1),
    }
    events = decide(_scores({393: 0.6, 394: 0.9}), lookup, threshold=0.5)
    assert len(events) == 1
    assert events[0].confidence == pytest.approx(0.9)


def test_decide_returns_events_sorted_by_confidence_desc():
    events = decide(_scores({DOORBELL: 0.6, 353: 0.9}),
                    {**LOOKUP, 353: EventSpec("knock", "敲门声", 2)},
                    threshold=0.5)
    assert [e.event_id for e in events] == ["knock", "doorbell"]


def test_decide_temporal_requires_min_frames():
    quiet = _scores({DOORBELL: 0.05})
    loud = _scores({DOORBELL: 0.9})
    frames = np.stack([loud, quiet, quiet, quiet, quiet])

    strict = decide_temporal(frames, LOOKUP, threshold=0.5, min_frames=2)
    assert strict == []

    lenient = decide_temporal(frames, LOOKUP, threshold=0.5, min_frames=1)
    assert len(lenient) == 1
    assert lenient[0].event_id == "doorbell"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_decision.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'decision'`

- [ ] **Step 3: 写实现**

`scripts/decision.py`:
```python
"""从 YAMNet 得分到听障提示事件的决策逻辑。

三级过滤：
  1) 帧级聚合——把 (n_frames, 521) 压成 (521,)
  2) 阈值判定——低于阈值的类别不产生事件
  3) 多帧一致性——事件需在足够多帧中重复出现，抑制瞬时误报
"""
from dataclasses import dataclass

import numpy as np

from event_mapping import (
    LEVEL_ALERT_1,
    LEVEL_ALERT_2,
    LEVEL_NAMES,
    EventSpec,
)


@dataclass(frozen=True)
class DetectedEvent:
    event_id: str
    name_cn: str
    level: int
    confidence: float
    audio_set_labels: tuple[str, ...] = ()

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES.get(self.level, "未知")

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "name_cn": self.name_cn,
            "level": self.level,
            "level_name": self.level_name,
            "confidence": round(float(self.confidence), 4),
            "audio_set_labels": list(self.audio_set_labels),
        }


def aggregate_frames(frame_scores: np.ndarray) -> np.ndarray:
    """把 (n_frames, n_classes) 的帧级得分沿帧轴取均值，返回 (n_classes,)。"""
    return frame_scores.mean(axis=0)


def _is_alertable(spec: EventSpec) -> bool:
    return spec.level in (LEVEL_ALERT_1, LEVEL_ALERT_2)


def decide(scores: np.ndarray, lookup: dict[int, EventSpec],
           threshold: float = 0.5,
           class_names: dict[int, str] | None = None) -> list[DetectedEvent]:
    """按阈值从聚合得分中选出事件，同一 event_id 取最高置信度。

    返回按置信度降序排列的事件列表。
    """
    class_names = class_names or {}

    # event_id → (最高置信度, 中文名, 等级, [触发该事件的类别索引])
    grouped: dict[str, tuple[float, str, int, list[int]]] = {}
    for index, score in enumerate(scores):
        spec = lookup.get(index)
        if spec is None or not _is_alertable(spec):
            continue
        value = float(score)
        if value < threshold:
            continue
        best, name_cn, level, indices = grouped.get(
            spec.event_id, (0.0, spec.name_cn, spec.level, [])
        )
        grouped[spec.event_id] = (
            max(best, value), name_cn, level, indices + [index]
        )

    events: list[DetectedEvent] = []
    for event_id, (confidence, name_cn, level, indices) in grouped.items():
        events.append(DetectedEvent(
            event_id=event_id,
            name_cn=name_cn,
            level=level,
            confidence=confidence,
            audio_set_labels=tuple(
                class_names.get(i, str(i)) for i in sorted(indices)
            ),
        ))

    events.sort(key=lambda e: e.confidence, reverse=True)
    return events


def decide_temporal(frame_scores: np.ndarray, lookup: dict[int, EventSpec],
                    threshold: float = 0.5, min_frames: int = 1,
                    class_names: dict[int, str] | None = None) -> list[DetectedEvent]:
    """在多帧上做一致性判定：类别需在至少 min_frames 帧中超过阈值才算命中。

    置信度取这些命中帧的均值。
    """
    class_names = class_names or {}

    # event_id → {帧号: 该帧中属于此事件的最高类别得分}
    per_event: dict[str, dict[int, float]] = {}
    specs: dict[str, EventSpec] = {}

    for frame in range(frame_scores.shape[0]):
        for index in np.where(frame_scores[frame] >= threshold)[0]:
            idx = int(index)
            spec = lookup.get(idx)
            if spec is None or not _is_alertable(spec):
                continue
            specs.setdefault(spec.event_id, spec)
            frame_scores_of_event = per_event.setdefault(spec.event_id, {})
            frame_scores_of_event[frame] = max(
                frame_scores_of_event.get(frame, 0.0),
                float(frame_scores[frame][idx]),
            )

    events: list[DetectedEvent] = []
    for event_id, frames in per_event.items():
        if len(frames) < min_frames:
            continue
        spec = specs[event_id]
        events.append(DetectedEvent(
            event_id=event_id,
            name_cn=spec.name_cn,
            level=spec.level,
            confidence=float(np.mean(list(frames.values()))),
            audio_set_labels=tuple(
                class_names.get(i, str(i))
                for i, s in lookup.items() if s.event_id == event_id
            ),
        ))

    events.sort(key=lambda e: e.confidence, reverse=True)
    return events
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_decision.py -v`
Expected: 8 passed

- [ ] **Step 5: 提交**

```bash
git add scripts/decision.py tests/test_decision.py
git commit -m "添加帧聚合、阈值判定与多帧去抖决策逻辑"
```

---

## Task 6: YAMNet ONNX 推理封装

**Files:**
- Create: `scripts/yamnet_runner.py`
- Create: `scripts/probe_model.py`
- Create: `tests/test_yamnet_runner.py`

- [ ] **Step 1: 探测模型的真实输入输出规格**

`scripts/probe_model.py`:
```python
"""打印 YAMNet ONNX 的输入输出名称与形状，并用合成信号跑一次推理。"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yamnet.onnx"


def main() -> int:
    session = onnxruntime.InferenceSession(
        str(MODEL_PATH), providers=["CPUExecutionProvider"]
    )

    print("=== 输入 ===")
    for inp in session.get_inputs():
        print(f"  name={inp.name!r} shape={inp.shape} type={inp.type}")

    print("=== 输出 ===")
    for out in session.get_outputs():
        print(f"  name={out.name!r} shape={out.shape} type={out.type}")

    input_name = session.get_inputs()[0].name
    for seconds in (1.0, 5.0):
        n = int(16000 * seconds)
        wave = (0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / 16000)).astype(np.float32)
        result = session.run(None, {input_name: wave})[0]
        print(f"{seconds}s 输入 -> 输出形状 {result.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run: `python scripts/probe_model.py`
Expected: 打印输入输出名称与形状，两行输出形状。**把实际形状记录下来**，用于确认下面的 `score()` 返回值语义。

**根据探测结果分支处理**：

- 若 1.0s 与 5.0s 两次推理都成功，且输出帧数随输入长度增加 → 模型接受变长输入，按下面的实现直接写。
- 若模型要求固定输入长度（例如只接受 15,600 或 160,000 采样点）→ 在 `score()` 内先把波形补零或截断到该长度，并把该约束写进代码注释与报告中。此时 `test_longer_audio_yields_more_frames` 会失败，**删除该测试并在报告中记录"模型仅支持固定长度输入"这一实测结论**——不要为了让测试通过而伪造行为。

- [ ] **Step 2: 写失败的测试**

`tests/test_yamnet_runner.py`:
```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from yamnet_runner import N_CLASSES, YamnetRunner

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yamnet.onnx"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="需要先运行 scripts/download_model.py"
)


@pytest.fixture(scope="module")
def runner():
    return YamnetRunner(MODEL_PATH)


def test_score_returns_one_row_per_frame(runner):
    wave = np.zeros(16000, dtype=np.float32)
    scores = runner.score(wave)
    assert scores.ndim == 2
    assert scores.shape[1] == N_CLASSES
    assert scores.shape[0] >= 1


def test_score_values_look_like_probabilities(runner):
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 0.1, 16000 * 2).astype(np.float32)
    scores = runner.score(wave)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_empty_audio_raises(runner):
    with pytest.raises(ValueError):
        runner.score(np.zeros(0, dtype=np.float32))


def test_longer_audio_yields_more_frames(runner):
    short = runner.score(np.zeros(16000, dtype=np.float32))
    long = runner.score(np.zeros(16000 * 5, dtype=np.float32))
    assert long.shape[0] > short.shape[0]
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_yamnet_runner.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'yamnet_runner'`

- [ ] **Step 4: 写实现**

`scripts/yamnet_runner.py`:
```python
"""YAMNet ONNX 推理封装。

模型输入为 16 kHz 单声道 float32 波形（预处理在计算图内部完成），
输出为 (n_frames, 521) 的 AudioSet 得分矩阵。
"""
from pathlib import Path

import numpy as np
import onnxruntime

N_CLASSES = 521


class YamnetRunner:
    """加载一次 ONNX 会话，可重复调用 score()。"""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"模型不存在: {self.model_path}\n请先运行: python scripts/download_model.py"
            )
        self.session = onnxruntime.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def score(self, waveform: np.ndarray) -> np.ndarray:
        """对 16 kHz 单声道波形推理，返回 (n_frames, 521) 的 float32 得分矩阵。"""
        wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if wave.size == 0:
            raise ValueError("输入波形为空，无法推理")

        output = self.session.run(None, {self.input_name: wave})[0]
        scores = np.asarray(output, dtype=np.float32)

        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        if scores.shape[1] != N_CLASSES:
            raise ValueError(
                f"模型输出类别数为 {scores.shape[1]}，预期 {N_CLASSES}。"
                f"请确认模型文件是否为 yamnetonnx/yamnet.onnx"
            )
        return scores
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_yamnet_runner.py -v`
Expected: 4 passed

- [ ] **Step 6: 提交**

```bash
git add scripts/yamnet_runner.py scripts/probe_model.py tests/test_yamnet_runner.py
git commit -m "添加 YAMNet ONNX 推理封装与模型探测脚本"
```

---

## Task 7: 合成测试信号

**Files:**
- Create: `scripts/make_test_signals.py`
- Create: `tests/test_make_test_signals.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_make_test_signals.py`:
```python
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_test_signals import SAMPLE_RATE, sine, silence, white_noise, write_wav


def test_sine_has_expected_length_and_rate():
    tone = sine(440.0, 1.0)
    assert len(tone) == SAMPLE_RATE
    assert tone.dtype == np.int16


def test_sine_frequency_is_correct():
    tone = sine(440.0, 1.0).astype(np.float32) / 32767
    spectrum = np.abs(np.fft.rfft(tone))
    peak_hz = np.fft.rfftfreq(len(tone), 1 / SAMPLE_RATE)[np.argmax(spectrum)]
    assert abs(peak_hz - 440.0) < 2.0


def test_silence_is_all_zero():
    assert np.all(silence(0.5) == 0)


def test_white_noise_is_deterministic_with_seed():
    assert np.array_equal(white_noise(0.5, seed=42), white_noise(0.5, seed=42))


def test_write_wav_creates_readable_file(tmp_path):
    path = tmp_path / "out.wav"
    write_wav(path, sine(440.0, 0.2))
    sr, data = wavfile.read(str(path))
    assert sr == SAMPLE_RATE
    assert len(data) == int(SAMPLE_RATE * 0.2)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_make_test_signals.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'make_test_signals'`

- [ ] **Step 3: 写实现**

`scripts/make_test_signals.py`:
```python
"""生成合成测试信号，用于验证推理管线连通性。

合成信号只能验证管线跑通与得分响应，不能用于评估识别准确率。
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

SAMPLE_RATE = 16000
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "signals"


def _n_samples(seconds: float) -> int:
    return int(SAMPLE_RATE * seconds)


def sine(freq_hz: float, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(_n_samples(seconds)) / SAMPLE_RATE
    wave = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return (wave * 32767).astype(np.int16)


def white_noise(seconds: float, amplitude: float = 0.2,
                seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wave = rng.uniform(-amplitude, amplitude, _n_samples(seconds))
    return (wave * 32767).astype(np.int16)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(_n_samples(seconds), dtype=np.int16)


def sweep(start_hz: float, end_hz: float, seconds: float) -> np.ndarray:
    t = np.arange(_n_samples(seconds)) / SAMPLE_RATE
    freq = np.linspace(start_hz, end_hz, len(t))
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    return (0.5 * np.sin(phase) * 32767).astype(np.int16)


def pulse_train(rate_hz: float, seconds: float,
                amplitude: float = 0.8) -> np.ndarray:
    """周期性脉冲串，模拟敲击/警报类瞬态信号。"""
    n = _n_samples(seconds)
    wave = np.zeros(n, dtype=np.float32)
    period = max(1, int(SAMPLE_RATE / rate_hz))
    burst = int(0.02 * SAMPLE_RATE)
    for start in range(0, n, period):
        end = min(start + burst, n)
        wave[start:end] = amplitude
    return (wave * 32767).astype(np.int16)


def write_wav(path: str | Path, audio: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), SAMPLE_RATE, audio)


def main() -> int:
    signals = {
        "tone_440hz.wav": sine(440.0, 3.0),
        "tone_1000hz.wav": sine(1000.0, 3.0),
        "sweep_100_8000.wav": sweep(100.0, 8000.0, 3.0),
        "white_noise.wav": white_noise(3.0),
        "silence.wav": silence(3.0),
        "pulses_4hz.wav": pulse_train(4.0, 5.0),
    }
    for name, audio in signals.items():
        write_wav(OUT_DIR / name, audio)
        print(f"已生成 {name}")
    print(f"输出目录: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_make_test_signals.py -v`
Expected: 5 passed

- [ ] **Step 5: 生成信号并跑通推理**

Run:
```bash
python scripts/make_test_signals.py
python -c "
import sys; sys.path.insert(0,'scripts')
from audio_io import load_audio
from yamnet_runner import YamnetRunner
from decision import aggregate_frames
r = YamnetRunner('models/yamnet.onnx')
for name in ['tone_440hz.wav','white_noise.wav','silence.wav']:
    wave, sr = load_audio(f'data/signals/{name}')
    s = aggregate_frames(r.score(wave))
    top = s.argsort()[-3:][::-1]
    print(name, [(int(i), round(float(s[i]),3)) for i in top])
"
```
Expected: 每个信号打印 3 个最高分类别索引与得分。**这一步证明"音频 → 得分"链路已通。**

- [ ] **Step 6: 提交**

```bash
git add scripts/make_test_signals.py tests/test_make_test_signals.py
git commit -m "添加合成测试信号生成脚本"
```

---

## Task 8: 主流程 detect.py

**Files:**
- Create: `scripts/detect.py`
- Create: `tests/test_detect.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_detect.py`:
```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_detect.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'detect'`

- [ ] **Step 3: 写实现**

`scripts/detect.py`:
```python
"""主流程：音频文件 → 事件 JSON。

用法:
    python scripts/detect.py <音频文件或目录> [--out 输出目录] \
        [--threshold 0.5] [--min-frames 1] [--json-only]
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_detect.py -v`
Expected: 5 passed

- [ ] **Step 5: 端到端跑通合成信号**

Run: `python scripts/detect.py data/signals`
Expected: 每个信号打印一行结果，`data/out/` 下生成对应 JSON。

- [ ] **Step 6: 提交**

```bash
git add scripts/detect.py tests/test_detect.py
git commit -m "添加音频到事件 JSON 的主流程"
```

---

## Task 9: ESC-50 抽样下载

**Files:**
- Create: `scripts/download_samples.py`
- Create: `tests/test_download_samples.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_download_samples.py`:
```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_download_samples.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'download_samples'`

- [ ] **Step 3: 写实现**

`scripts/download_samples.py`:
```python
"""从 ESC-50 抽取与听障提示相关的片段。

ESC-50 整体为 CC BY-NC 3.0（署名—非商业），仅可用于教学，不得商用或再分发。
引用: Piczak, K. J. ESC: Dataset for Environmental Sound Classification. ACM MM 2015.

单个片段可独立下载，无需拉取全量包。
"""
import argparse
import csv
import io
import socket
import sys
import urllib.request
from contextlib import contextmanager
from pathlib import Path

BASE_AUDIO_URL = (
    "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio"
)
META_URL = (
    "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "samples"

# 与听障提示相关的类别；映射关系见报告第四、五章
HEARING_RELEVANT_CATEGORIES = [
    "siren",
    "car_horn",
    "crying_baby",
    "glass_breaking",
    "clock_alarm",
    "door_wood_knock",
    "door_wood_creaks",
    "dog",
    "fireworks",
    "thunderstorm",
    "church_bells",
    "vacuum_cleaner",
    "washing_machine",
]

EXPECTED_SAMPLE_BYTES = 441044


@contextmanager
def _force_ipv4():
    """下载期间限制 DNS 解析只返回 IPv4。

    本机到 raw.githubusercontent.com 的 4 个 IPv6 地址中有 3 个不可达
    （实测超时），而 urllib 没有实现 Happy Eyeballs（RFC 8305），
    会按 DNS 返回顺序逐个尝试，撞上不可达的 IPv6 就卡到超时。
    表现为间歇性：落到可用地址时 0.1s 完成，落到死地址则约 40s 后失败。
    curl 实现了地址竞速，因此同一 URL 用 curl 始终正常。
    """
    original = socket.getaddrinfo

    def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return original(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = ipv4_only
    try:
        yield
    finally:
        socket.getaddrinfo = original


def fetch_text(url: str, timeout: int = 60) -> str:
    with _force_ipv4():
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")


def parse_metadata(text: str) -> list[dict]:
    """解析 ESC-50 标注文本。

    注意：CSV 的 filename 列自带 .wav 后缀（如 1-100032-A-0.wav），
    这里统一去掉后缀。全流程用「无后缀的文件名」标识一个片段，
    拼接 URL 与本地路径时再补 .wav；不去掉会导致重复后缀而 404。
    """
    rows = list(csv.DictReader(io.StringIO(text)))
    for row in rows:
        row["filename"] = Path(row["filename"]).stem
    return rows


def load_metadata(url: str = META_URL) -> list[dict]:
    return parse_metadata(fetch_text(url))


def pick_filenames(rows: list[dict], categories: list[str],
                   per_category: int) -> list[dict]:
    """每类取 per_category 条，尽量分散在不同 fold，结果按文件名排序。

    确定性选取，保证重复运行得到同一批样本。
    """
    picked: list[dict] = []
    for category in categories:
        candidates = sorted(
            (r for r in rows if r["category"] == category),
            key=lambda r: (r["fold"], r["filename"]),
        )
        by_fold: dict[str, list[dict]] = {}
        for row in candidates:
            by_fold.setdefault(row["fold"], []).append(row)

        chosen: list[dict] = []
        fold_keys = sorted(by_fold)
        while len(chosen) < per_category:
            progressed = False
            for fold in fold_keys:
                if len(chosen) >= per_category:
                    break
                bucket = by_fold[fold]
                if bucket:
                    chosen.append(bucket.pop(0))
                    progressed = True
            if not progressed:
                break

        picked.extend(chosen)

    return sorted(picked, key=lambda r: r["filename"])


def download_sample(filename: str, dest_dir: Path,
                    retries: int = 3) -> Path:
    """下载单个 wav。逐条重试，失败则抛错，不静默跳过。"""
    dest = dest_dir / f"{filename}.wav"
    if dest.exists() and dest.stat().st_size == EXPECTED_SAMPLE_BYTES:
        return dest
    url = f"{BASE_AUDIO_URL}/{filename}.wav"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with _force_ipv4():
                with urllib.request.urlopen(url, timeout=60) as resp:
                    data = resp.read()
            if len(data) != EXPECTED_SAMPLE_BYTES:
                raise RuntimeError(
                    f"大小异常: {len(data)} 字节，预期 {EXPECTED_SAMPLE_BYTES}"
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return dest
        except Exception as exc:
            last_error = exc
            print(f"  第 {attempt}/{retries} 次失败: {filename} ({exc})")
    raise RuntimeError(f"下载失败: {url}") from last_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 ESC-50 抽样片段")
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--out", type=Path, default=SAMPLES_DIR)
    args = parser.parse_args(argv)

    rows = load_metadata()
    picked = pick_filenames(rows, HEARING_RELEVANT_CATEGORIES,
                            args.per_category)
    print(f"计划下载 {len(picked)} 条，覆盖 "
          f"{len({r['category'] for r in picked})} 个类别")

    failures: list[str] = []
    for row in picked:
        try:
            download_sample(row["filename"], args.out)
        except Exception as exc:
            failures.append(f"{row['filename']} ({row['category']}): {exc}")

    manifest = args.out / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "category", "fold"])
        writer.writeheader()
        for row in picked:
            if (args.out / f"{row['filename']}.wav").exists():
                writer.writerow({k: row[k] for k in
                                 ("filename", "category", "fold")})

    print(f"完成。清单: {manifest}")
    if failures:
        print(f"\n以下 {len(failures)} 条下载失败:", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_download_samples.py -v`
Expected: 5 passed

- [ ] **Step 5: 实际下载抽样**

Run: `python scripts/download_samples.py --per-category 5`
Expected: 计划下载 65 条，覆盖 13 个类别；完成后生成 `data/samples/manifest.csv`。（约 28 MB，视网络耗时数分钟。）

- [ ] **Step 6: 提交**

```bash
git add scripts/download_samples.py tests/test_download_samples.py
git commit -m "添加 ESC-50 听障相关类别抽样下载脚本"
```

---

## Task 10: 评估与指标

**Files:**
- Create: `scripts/evaluate.py`
- Create: `tests/test_evaluate.py`

**指标口径**：统计"ESC-50 类别 → 映射后听障事件的命中情况"，即零样本预测是否落进该类别应有的听障事件族。**不是**标准 50 类分类准确率，报告须写明口径。

- [ ] **Step 1: 写失败的测试**

`tests/test_evaluate.py`:
```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'evaluate'`

- [ ] **Step 3: 写实现**

`scripts/evaluate.py`:
```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: 8 passed

- [ ] **Step 5: 用默认阈值跑一次评估，确认链路通**

Run: `python scripts/evaluate.py --threshold 0.2`
Expected: 逐条打印预期/预测与对错，末尾给出事件级命中率，生成 `data/out/metrics.csv` 与 `metrics_summary.json`。

**注意**：YAMNet 的 521 类 sigmoid 得分整体偏低，默认阈值 0.5 很可能命中率接近 0。这不是代码故障，是模型得分分布的正常表现——下面的阈值扫描就是为了量化这一点。若 Step 5 命中率极低，属预期现象，继续 Step 6。

- [ ] **Step 6: 阈值与去抖参数扫描**

Run:
```bash
for t in 0.05 0.10 0.20 0.30 0.50; do
  python scripts/evaluate.py --threshold $t --out "data/out/thr_$t"
done
for m in 1 2 3; do
  python scripts/evaluate.py --threshold 0.10 --min-frames $m --out "data/out/mf_$m"
done
```
Expected: 8 个目录各有一份 `metrics_summary.json`。

- [ ] **Step 7: 汇总扫描结果成表**

Run:
```bash
python -c "
import json, glob
print(f'{\"参数\":<18}{\"命中率\":>8}{\"命中/总数\":>12}')
for p in sorted(glob.glob('data/out/thr_*/metrics_summary.json')):
    s = json.load(open(p, encoding='utf-8'))
    print(f'{p.split(chr(92))[-2]:<18}{s[\"hit_rate\"]:>8.1%}{s[\"hits\"]:>7}/{s[\"total\"]:<4}')
for p in sorted(glob.glob('data/out/mf_*/metrics_summary.json')):
    s = json.load(open(p, encoding='utf-8'))
    print(f'{p.split(chr(92))[-2]:<18}{s[\"hit_rate\"]:>8.1%}{s[\"hits\"]:>7}/{s[\"total\"]:<4}')
"
```
Expected: 一张参数—命中率对照表。**这是报告第五章"阈值与去抖参数对比"的核心数据。** 若脚本因路径分隔符报错，改用逐目录 `python -c` 读取或直接查看各 `metrics_summary.json`。

- [ ] **Step 8: 确定并在报告中说明最终参数**

从扫描结果中选一组作为推荐参数，**必须在报告中写明选择依据**（是偏向召回还是偏向抑制误报），不得只给结论不给理由。

- [ ] **Step 9: 生成报告图表**

**Files:**
- Create: `scripts/plot_results.py`
- Create: `tests/test_plot_results.py`

先写失败的测试 `tests/test_plot_results.py`:

```python
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
```

运行 `python -m pytest tests/test_plot_results.py -v` 确认失败（ModuleNotFoundError）。

然后写实现 `scripts/plot_results.py`:

```python
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
```

运行 `python -m pytest tests/test_plot_results.py -v` 确认通过（5 passed）。

然后实际出图：`python scripts/plot_results.py`
Expected: 打印图表路径与各参数命中率，`data/out/threshold_sweep.png` 生成。

- [ ] **Step 10: 提交**

```bash
git add scripts/evaluate.py tests/test_evaluate.py scripts/plot_results.py tests/test_plot_results.py
git commit -m "添加评估脚本与报告图表生成

- evaluate.py: ESC-50 事件级命中率统计
- plot_results.py: 阈值与去抖参数的对比图表"
```

---

## Task 11: Coze 工作流搭建与演示

这一步在 Coze 网页端操作，产出物是截图与粘贴记录，供报告第四章场景 3 使用。

**Files:**
- Create: `docs/coze-paste-samples.md`（粘贴用的 JSON 与结果记录）

- [ ] **Step 1: 生成粘贴用 JSON**

Run:
```bash
python scripts/detect.py data/samples --out data/out/samples_json
ls data/out/samples_json | head
```
Expected: 每个样本一份 JSON。

- [ ] **Step 2: 挑 3 份代表性 JSON**

挑一份一级事件（如 siren/glass_breaking）、一份二级事件（如 doorbell 类）、一份空事件。Run: `ls data/out/samples_json`

- [ ] **Step 3: 在 Coze 搭建工作流**

在 Coze 网页端新建工作流，结构为：开始节点 → 大模型节点 → 结束节点。
大模型节点的系统提示词粘贴 `docs/superpowers/specs/2026-09-18-hearing-accessibility-sed-design.md` 第 7.3 节的**提示词全文**（代码块内内容）。
温度设为 0.3。

- [ ] **Step 4: 依次粘贴 3 份 JSON 并记录输出**

把 Step 2 挑出的三份 JSON 分别粘贴进工作流输入，记录每次输出的中文提示文案。

- [ ] **Step 5: 把 JSON 与输出写入 `docs/coze-paste-samples.md`**

格式：
```markdown
# Coze 工作流粘贴样本与输出记录

## 样本 1：一级事件

### 粘贴的 JSON
```json
（此处粘贴实际 JSON）
```

### Coze 输出
（此处粘贴实际输出文案）

### 观察
（记录模型是否遵守了"不推测未出现声音""低置信度用可能"等规则）
```

- [ ] **Step 6: 截图**

工作流编排界面、提示词配置界面、每次运行的输入输出界面，共若干张，存入 `docs/images/`，供报告插图。

- [ ] **Step 7: 提交**

```bash
git add docs/coze-paste-samples.md docs/images
git commit -m "记录 Coze 工作流搭建过程与粘贴样本输出"
```

---

## Task 12: 撰写实验报告

**Files:**
- Create: `docs/听障无障碍环境声音提示智能体实验报告.docx`

- [ ] **Step 1: 读取模板结构**

用 docx skill 读取 `音频智能体实验报告.bak.docx`，确认八节结构与各场景五段式的实际样式（标题级别、字体、编号方式），新报告必须沿用同一套样式。

- [ ] **Step 2: 按模板结构撰写全文**

项目名称：听障无障碍环境声音提示智能体——基于 YAMNet 零样本声学事件检测与 Coze 智能体

八节内容来源：
- 一、实验目的：设计文档第 1 节
- 二、实验原理：设计文档第 3 节 + YAMNet/AudioSet/SED 原理
- 三、实验环境：设计文档第 2、3 节，含本机网络约束表
- 四、实验内容与方案设计：三个场景，每个五段式
  - 场景 1：YAMNet 零样本环境声音识别 → Task 6、7 的产出
  - 场景 2：AudioSet→听障事件映射与分级决策 → Task 4、5 的产出
  - 场景 3：Coze 智能体分级提示文案生成 → Task 11 的产出
- 五、实验结果与分析：Task 4 Step 5 的覆盖统计、Task 10 Step 5 的首次命中率、Task 10 Step 6–8 的阈值与去抖参数扫描表
- 六、实验伦理与风险讨论：误报漏报的安全后果、持续监听隐私、**ESC-50 CC BY-NC 3.0 署名与非商用声明**、AudioSet 本体偏见、责任边界
- 七、实验提交要求：同模板
- 八、思考题：同模板

- [ ] **Step 3: 插入图表**

至少包含：系统架构图、映射分级表、覆盖率统计表、混淆矩阵（由 `build_confusion` 产出）、命中率对比表、`data/out/threshold_sweep.png`（由 `scripts/plot_results.py` 生成）、若干 Coze 截图。

- [ ] **Step 4: 校验交付物**

Run:
```bash
python -c "
from docx import Document
d = Document('docs/听障无障碍环境声音提示智能体实验报告.docx')
paras = [p.text for p in d.paragraphs if p.text.strip()]
for section in ['一、实验目的','二、实验原理','三、实验环境','四、实验内容与方案设计','五、实验结果与分析','六、实验伦理与风险讨论','七、实验提交要求','八、思考题']:
    assert any(section in p for p in paras), f'缺少章节: {section}'
print('八节结构完整，共', len(paras), '段')
"
```
Expected: `八节结构完整，共 N 段`

- [ ] **Step 5: 提交**

```bash
git add "docs/听障无障碍环境声音提示智能体实验报告.docx"
git commit -m "完成听障无障碍环境声音提示智能体实验报告"
```

---

## 验证清单（声称完成前必须逐项确认）

- [ ] `python -m pytest tests/ -v` 全部通过
- [ ] `python scripts/download_model.py` 输出 521 类
- [ ] `python scripts/detect.py data/signals` 生成 JSON 且无异常
- [ ] `python scripts/evaluate.py` 产出 `metrics.csv` 与 `metrics_summary.json`
- [ ] 报告八节结构完整，含真实实验数据而非占位
- [ ] 报告第六章含 ESC-50 许可证声明与署名
- [ ] 报告中所有非自测数据均标注了来源

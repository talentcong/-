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


def download(url: str, dest: Path) -> None:
    """下载 url 到 dest。已存在且非空则跳过。"""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"已存在，跳过: {dest.name}")
        return
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
    tmp.replace(dest)


def parse_class_map(path: Path) -> dict[int, str]:
    """解析 yamnet_class_map.csv，返回 {类别索引: 显示名}。"""
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["index"]): row["display_name"] for row in csv.DictReader(f)}


def main() -> int:
    download(MODEL_URL, MODEL_PATH)
    download(CLASS_MAP_URL, CLASS_MAP_PATH)

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

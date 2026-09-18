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

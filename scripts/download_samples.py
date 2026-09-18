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

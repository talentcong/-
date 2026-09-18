import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_model import parse_class_map, MODEL_URL, CLASS_MAP_URL


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

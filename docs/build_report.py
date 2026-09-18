"""生成实验报告 .docx。

沿用 音频智能体实验报告.docx 模板的段落格式：
无命名样式，每段为 <w:rFonts w:hint="eastAsia"/> 的素文本段落。
页面设置沿用模板的 A4 + 1800 左右边距。
"""
import io
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "音频智能体实验报告.docx"
OUT = ROOT / "docs" / "听障无障碍环境声音提示智能体实验报告.docx"
CHART = ROOT / "data" / "out" / "threshold_sweep.png"

W_NS = (
    'xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:o="urn:schemas-microsoft-com:office:office" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:v="urn:schemas-microsoft-com:vml" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
    'xmlns:w10="urn:schemas-microsoft-com:office:word" '
    'xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml" '
    'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
    'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
    'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
    'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
    'mc:Ignorable="w14 w15 wp14"'
)

SECTPR = (
    '<w:sectPr>'
    '<w:pgSz w:w="11906" w:h="16838"/>'
    '<w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" '
    'w:header="851" w:footer="992" w:gutter="0"/>'
    '<w:cols w:space="425" w:num="1"/>'
    '<w:docGrid w:type="lines" w:linePitch="312" w:charSpace="0"/>'
    '</w:sectPr>'
)


def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def para(text: str) -> str:
    """素文本段落；空字符串产出空段落。"""
    if not text:
        return "<w:p/>"
    return (
        '<w:p><w:pPr><w:rPr><w:rFonts w:hint="eastAsia"/></w:rPr></w:pPr>'
        f'<w:r><w:rPr><w:rFonts w:hint="eastAsia"/></w:rPr>'
        f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'
    )


def image_para(rel_id: str, name: str) -> str:
    """按页面可用宽度等比缩放图片。

    页面 A4 宽 11906 DXA，左右边距各 1800，可用宽度 8306 DXA。
    取 95% 留少量余量，避免图片溢出到页边距之外。
    """
    import struct

    data = CHART.read_bytes()
    px_w, px_h = struct.unpack(">II", data[16:24])

    usable_dxa = 11906 - 1800 * 2
    cx = int(usable_dxa * 0.95 * 635)  # 1 DXA = 635 EMU
    cy = int(cx * px_h / px_w)
    return _image_xml(rel_id, cx, cy, name)


def _image_xml(rel_id: str, cx: int, cy: int, name: str) -> str:
    return (
        '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="1" name="{esc(name)}"/>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:nvPicPr><pic:cNvPr id="1" name="' + esc(name) + '"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
    )


def build_body() -> str:
    import report_content as C

    parts = []
    for block in C.CONTENT:
        kind = block[0]
        if kind == "p":
            parts.append(para(block[1]))
        elif kind == "blank":
            parts.append("<w:p/>")
        elif kind == "image":
            parts.append(image_para("rId100", block[1]))
    parts.append(SECTPR)
    return "".join(parts)


def main() -> int:
    if not CHART.exists():
        raise SystemExit(f"缺少图表: {CHART}，请先运行 scripts/plot_results.py")

    work = ROOT / "_report_build"
    if work.exists():
        shutil.rmtree(work)
    with zipfile.ZipFile(TEMPLATE) as z:
        z.extractall(work)

    doc = work / "word" / "document.xml"
    xml = doc.read_text(encoding="utf-8")
    head = xml[: xml.index("<w:body>") + len("<w:body>")]
    doc.write_text(
        head + build_body() + "</w:body></w:document>", encoding="utf-8"
    )

    media = work / "word" / "media"
    media.mkdir(exist_ok=True)
    shutil.copy(CHART, media / "threshold_sweep.png")

    rels = work / "word" / "_rels" / "document.xml.rels"
    rels_xml = rels.read_text(encoding="utf-8")
    if "rId100" not in rels_xml:
        rels_xml = rels_xml.replace(
            "</Relationships>",
            '<Relationship Id="rId100" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/image" Target="media/threshold_sweep.png"/>'
            "</Relationships>",
        )
        rels.write_text(rels_xml, encoding="utf-8")

    ct = work / "[Content_Types].xml"
    ct_xml = ct.read_text(encoding="utf-8")
    if 'Extension="png"' not in ct_xml:
        ct_xml = ct_xml.replace(
            "<Default Extension=",
            '<Default Extension="png" ContentType="image/png"/><Default Extension=',
            1,
        )
        ct.write_text(ct_xml, encoding="utf-8")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(work.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(work).as_posix())
    shutil.rmtree(work)
    print(f"已生成: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert a PDF file into an editable HWPX (한/글) document.

Each line of PDF text becomes a real, editable HWPX paragraph/run: font
size, bold, italic and colour are carried over from the PDF, and the
line's original position is approximated with a left indent (horizontal)
and space-before (vertical) computed from the PDF coordinates. Images are
re-embedded as floating pictures anchored to their original page position.

This is a best-effort layout reconstruction, not a pixel-perfect one:
HWPX paragraphs are a flowing text model, not a collection of absolutely
positioned text boxes, so very dense, multi-column, or rotated layouts
will not match exactly. See the README section at the bottom of this
file (or `python pdf_to_hwpx.py --help`) for details.

Usage:
    python pdf_to_hwpx.py input.pdf output.hwpx [--dpi 150] [--max-pages N]

Requires:
    pip install pymupdf python-hwpx
"""

from __future__ import annotations

import argparse
import logging
import sys

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF가 필요합니다. 먼저 'pip install pymupdf'를 실행하세요.")

try:
    from hwpx.document import HwpxDocument
except ImportError:
    sys.exit("python-hwpx가 필요합니다. 먼저 'pip install python-hwpx'를 실행하세요.")

logging.getLogger("hwpx").setLevel(logging.ERROR)

PT_TO_MM = 25.4 / 72.0
PT_TO_HWPUNIT = 100  # HWPUNIT = 1/7200 inch; 1pt = 7200/72 = 100 HWPUNIT

FONT_ITALIC_FLAG = 1 << 1
FONT_BOLD_FLAG = 1 << 4


def pt_to_hwpunit(value: float) -> int:
    return round(value * PT_TO_HWPUNIT)


def span_is_bold(span: dict) -> bool:
    if span.get("flags", 0) & FONT_BOLD_FLAG:
        return True
    return "bold" in span.get("font", "").lower()


def span_is_italic(span: dict) -> bool:
    if span.get("flags", 0) & FONT_ITALIC_FLAG:
        return True
    font_name = span.get("font", "").lower()
    return "italic" in font_name or "oblique" in font_name


def span_color(span: dict) -> str:
    value = span.get("color", 0)
    return f"#{value & 0xFFFFFF:06X}"


def extract_page_lines(page: "fitz.Page") -> list[dict]:
    """Return text lines on *page*, sorted into a top-to-bottom reading order.

    Multi-column pages are not reflowed into columns; lines are simply
    ordered by vertical position, then horizontal position.
    """
    raw = page.get_text("dict")
    lines = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # 0 == text block
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            lines.append({"bbox": line["bbox"], "spans": spans})
    lines.sort(key=lambda ln: (round(ln["bbox"][1], 1), ln["bbox"][0]))
    return lines


def extract_page_images(page: "fitz.Page", dpi: int) -> list[dict]:
    """Render each placed image's own page region as a standalone PNG.

    Rendering the placement rect (rather than extracting the raw XObject)
    keeps this robust across colour spaces, soft masks and clipped/rotated
    placements, at the cost of re-rasterising at a fixed DPI.
    """
    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    seen_bboxes = set()
    for info in page.get_image_info():
        bbox = info.get("bbox")
        if not bbox:
            continue
        key = tuple(round(v, 1) for v in bbox)
        if key in seen_bboxes:
            continue
        seen_bboxes.add(key)
        rect = fitz.Rect(bbox)
        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            continue
        try:
            pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
            png_bytes = pix.tobytes("png")
        except Exception:
            continue
        images.append({"bbox": bbox, "png": png_bytes})
    return images


def build_hwpx(pdf_path: str, output_path: str, *, dpi: int = 150, max_pages: int | None = None) -> None:
    src = fitz.open(pdf_path)
    try:
        if src.page_count == 0:
            raise ValueError("PDF에 페이지가 없습니다.")

        page_count = src.page_count if max_pages is None else min(max_pages, src.page_count)
        first_page = src[0]

        with HwpxDocument.new() as doc:
            doc.set_page_setup(
                width_mm=first_page.rect.width * PT_TO_MM,
                height_mm=first_page.rect.height * PT_TO_MM,
                margin_left_mm=0,
                margin_right_mm=0,
                margin_top_mm=0,
                margin_bottom_mm=0,
                header_margin_mm=0,
                footer_margin_mm=0,
                gutter_mm=0,
            )

            first_paragraph = doc.paragraphs[0]
            warned_size_mismatch = False

            for page_index in range(page_count):
                page = src[page_index]

                if not warned_size_mismatch and (
                    abs(page.rect.width - first_page.rect.width) > 1.0
                    or abs(page.rect.height - first_page.rect.height) > 1.0
                ):
                    print(
                        f"[경고] {page_index + 1}페이지 크기가 1페이지와 달라 "
                        "전체 문서에는 1페이지 크기를 사용합니다.",
                        file=sys.stderr,
                    )
                    warned_size_mismatch = True

                lines = extract_page_lines(page)
                images = extract_page_images(page, dpi)

                page_anchor_paragraph = None
                prev_bottom_pt = 0.0
                is_first_paragraph_on_page = True

                for line in lines:
                    x0, y0, x1, y1 = line["bbox"]
                    gap_pt = max(0.0, y0 - prev_bottom_pt)
                    prev_bottom_pt = y1

                    if page_index == 0 and is_first_paragraph_on_page:
                        paragraph = first_paragraph
                    else:
                        paragraph = doc.add_paragraph("", include_run=False)

                    if page_index > 0 and is_first_paragraph_on_page:
                        paragraph.element.set("pageBreak", "1")

                    is_first_paragraph_on_page = False
                    if page_anchor_paragraph is None:
                        page_anchor_paragraph = paragraph

                    for span in line["spans"]:
                        text = span.get("text", "")
                        if not text:
                            continue
                        paragraph.add_run(
                            text,
                            bold=span_is_bold(span),
                            italic=span_is_italic(span),
                            color=span_color(span),
                            size=round(span.get("size", 10.0), 1),
                        )

                    para_index = doc.paragraphs.index(paragraph)
                    doc.set_paragraph_format(
                        paragraph_index=para_index,
                        alignment="LEFT",
                        line_spacing_percent=100,
                        indent_left_mm=max(0.0, x0) * PT_TO_MM,
                        spacing_before_pt=gap_pt,
                    )

                if images:
                    if page_anchor_paragraph is None:
                        page_anchor_paragraph = doc.add_paragraph("", include_run=False)
                        if page_index > 0 and is_first_paragraph_on_page:
                            page_anchor_paragraph.element.set("pageBreak", "1")
                        is_first_paragraph_on_page = False

                    for image in images:
                        x0, y0, x1, y1 = image["bbox"]
                        item_id = doc.add_image(image["png"], "png")
                        page_anchor_paragraph.add_picture(
                            item_id,
                            width=pt_to_hwpunit(x1 - x0),
                            height=pt_to_hwpunit(y1 - y0),
                            treat_as_char=False,
                            pos_overrides={
                                "horzRelTo": "PAPER",
                                "vertRelTo": "PAPER",
                                "horzAlign": "LEFT",
                                "vertAlign": "TOP",
                                "horzOffset": pt_to_hwpunit(x0),
                                "vertOffset": pt_to_hwpunit(y0),
                            },
                        )

                print(f"  {page_index + 1}/{page_count} 페이지 처리 완료 "
                      f"(텍스트 줄 {len(lines)}개, 이미지 {len(images)}개)")

            doc.save_to_path(output_path)
    finally:
        src.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF를 HWPX로 변환합니다 (레이아웃 최대한 보존).")
    parser.add_argument("input_pdf", help="입력 PDF 파일 경로")
    parser.add_argument("output_hwpx", help="출력 HWPX 파일 경로")
    parser.add_argument("--dpi", type=int, default=150, help="이미지 재렌더링 해상도 (기본 150)")
    parser.add_argument("--max-pages", type=int, default=None, help="변환할 최대 페이지 수 (기본: 전체)")
    args = parser.parse_args()

    print(f"'{args.input_pdf}' 변환을 시작합니다...")
    build_hwpx(args.input_pdf, args.output_hwpx, dpi=args.dpi, max_pages=args.max_pages)
    print(f"완료: '{args.output_hwpx}'")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 알아두어야 할 한계 (README)
# ---------------------------------------------------------------------------
# - HWPX 문단은 절대좌표 텍스트 상자가 아니라 "흐르는" 텍스트 모델입니다. 이 스크립트는
#   PDF의 각 텍스트 줄을 별도 문단으로 만들고, 왼쪽 들여쓰기(가로 위치)와 이전 줄과의
#   간격(세로 위치)으로 원래 위치를 근사합니다. 완벽한 픽셀 단위 일치는 아닙니다.
# - PDF에 쓰인 원본 폰트는 그대로 옮겨지지 않고, 한/글 기본 폰트로 대체됩니다. 글자
#   폭이 달라지므로 줄바꿈 위치가 PDF와 미세하게 어긋날 수 있습니다.
# - 다단(multi-column) 레이아웃은 열을 인식하지 못하고 위→아래 순서로만 배치됩니다.
# - 회전된 텍스트, 벡터 도형(선/사각형 등)은 변환되지 않습니다. 이미지는 해당 영역을
#   비트맵으로 다시 렌더링해 원래 위치에 삽입합니다.
# - 페이지 크기가 페이지마다 다른 PDF는 1페이지 크기로 통일됩니다.

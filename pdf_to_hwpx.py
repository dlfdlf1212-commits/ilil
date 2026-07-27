#!/usr/bin/env python3
"""Convert a PDF file into an editable HWPX (한/글) document.

Each line of PDF text becomes a real, editable HWPX paragraph/run: font
size, bold, italic and colour are carried over from the PDF, and the
line's original position is approximated with a left indent (horizontal)
and space-before (vertical) computed from the PDF coordinates. Detected
tables become real HWPX tables (<hp:tbl>), not loose text. Images are
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

# A table nested inside another detected table's cell is reported as its
# own separate table by find_tables(); if it overlaps an already-kept
# table by more than this fraction of its own area, treat it as a
# duplicate/nested detection and drop it rather than rendering the same
# content twice.
NESTED_TABLE_OVERLAP_THRESHOLD = 0.8


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


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def extract_page_tables(page: "fitz.Page") -> list:
    """Return detected tables for *page*, with nested/duplicate detections dropped.

    ``page.find_tables()`` sometimes reports a table *and* the smaller
    tables nested inside its cells as separate results. Keeping both would
    render the same content twice, so only the largest (outermost) table
    in each overlapping cluster is kept; anything nested inside it stays
    as flattened text within that outer table's cell.
    """
    try:
        found = list(page.find_tables().tables)
    except Exception:
        return []

    def has_text(t) -> bool:
        try:
            grid = t.extract()
        except Exception:
            return False
        return any((cell or "").strip() for row in grid for cell in row)

    found = [
        t for t in found
        if t.row_count > 0 and t.col_count > 0 and _bbox_area(t.bbox) > 0 and has_text(t)
    ]
    found.sort(key=lambda t: _bbox_area(t.bbox), reverse=True)

    kept = []
    for table in found:
        area = _bbox_area(table.bbox)
        if any(_bbox_overlap_area(table.bbox, k.bbox) / area > NESTED_TABLE_OVERLAP_THRESHOLD for k in kept):
            continue
        kept.append(table)

    kept.sort(key=lambda t: (round(t.bbox[1], 1), t.bbox[0]))
    return kept


def extract_page_lines(page: "fitz.Page", exclude_bboxes: list[tuple[float, float, float, float]] = ()) -> list[dict]:
    """Return text lines on *page*, sorted into a top-to-bottom reading order.

    Lines whose centre falls inside one of *exclude_bboxes* (detected
    tables, handled separately) are skipped so table text is not
    duplicated as loose paragraphs. Multi-column pages are not reflowed
    into columns; lines are simply ordered by vertical position, then
    horizontal position.
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
            x0, y0, x1, y1 = line["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for bx0, by0, bx1, by1 in exclude_bboxes):
                continue
            lines.append({"bbox": line["bbox"], "spans": spans})
    lines.sort(key=lambda ln: (round(ln["bbox"][1], 1), ln["bbox"][0]))
    return lines


def _merge_boxes(boxes: list[tuple[float, float, float, float]], gap: float = 2.0) -> list[tuple[float, float, float, float]]:
    """Union any boxes that touch or overlap (within *gap* points) into one."""
    pending = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        merged: list[list[float]] = []
        while pending:
            box = pending.pop()
            i = 0
            while i < len(pending):
                other = pending[i]
                separated = (
                    box[2] + gap < other[0] or other[2] + gap < box[0]
                    or box[3] + gap < other[1] or other[3] + gap < box[1]
                )
                if not separated:
                    box = [min(box[0], other[0]), min(box[1], other[1]),
                           max(box[2], other[2]), max(box[3], other[3])]
                    pending.pop(i)
                    changed = True
                else:
                    i += 1
            merged.append(box)
        pending = merged
    return [tuple(b) for b in pending]


def extract_page_vector_regions(
    page: "fitz.Page",
    dpi: int,
    *,
    text_bboxes: list[tuple[float, float, float, float]] = (),
    max_area_fraction: float = 0.35,
) -> list[dict]:
    """Rasterise vector-drawn decoration (coloured bars, boxes, gradients, ...).

    PDF viewers render lines/rectangles/gradients as vector paths, which
    this converter otherwise ignores entirely (only text and raster images
    are picked up). Nearby paths are merged into bounding regions and each
    region is re-rendered as a PNG so simple decorations (divider bars,
    shaded boxes, etc.) survive the conversion.

    Regions that overlap any text line are dropped rather than rasterised,
    since that text is already being reproduced as real, editable HWPX
    text elsewhere — baking it into a picture too would duplicate it.
    Very large regions (more than *max_area_fraction* of the page) are
    also dropped so an incidental full-page background fill doesn't turn
    into one giant image covering everything else.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []

    page_rect = page.rect
    boxes = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None or (d.get("fill") is None and d.get("color") is None):
            continue
        clipped = fitz.Rect(rect) & page_rect
        if clipped.is_empty or clipped.width <= 0 or clipped.height <= 0:
            continue
        boxes.append((clipped.x0, clipped.y0, clipped.x1, clipped.y1))
    if not boxes:
        return []

    page_area = page_rect.width * page_rect.height
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    regions = []
    for bbox in _merge_boxes(boxes):
        x0, y0, x1, y1 = bbox
        if (x1 - x0) <= 0 or (y1 - y0) <= 0:
            continue
        region_area = (x1 - x0) * (y1 - y0)
        if region_area > page_area * max_area_fraction:
            continue
        # Ignore grazing overlaps: a text line's bbox from get_text("dict")
        # includes generous ascent/descent padding (especially for CJK
        # fonts) that extends past the visible glyph ink, so shrink it
        # vertically before comparing — only a real, visually meaningful
        # overlap (text actually sitting inside the region) should drop it.
        def _shrunk(tb: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
            tx0, ty0, tx1, ty1 = tb
            inset = min(3.0, (ty1 - ty0) * 0.25)
            return (tx0, ty0 + inset, tx1, ty1 - inset)

        if any(
            _bbox_overlap_area(bbox, shrunk) > 0.02 * min(region_area, _bbox_area(shrunk))
            for tb in text_bboxes
            if _bbox_area(shrunk := _shrunk(tb)) > 0
        ):
            continue
        try:
            pix = page.get_pixmap(matrix=matrix, clip=fitz.Rect(bbox), alpha=True)
            png_bytes = pix.tobytes("png")
        except Exception:
            continue
        regions.append({"bbox": bbox, "png": png_bytes, "behind_text": True})
    return regions


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


def add_table_block(doc: "HwpxDocument", table) -> "object":
    """Insert *table* (a fitz.table.Table) as a real HWPX table and return it."""
    x0, y0, x1, y1 = table.bbox
    hwpx_table = doc.add_table(
        table.row_count,
        table.col_count,
        width=pt_to_hwpunit(x1 - x0),
        height=pt_to_hwpunit(y1 - y0),
    )

    first_row_cells = table.rows[0].cells if table.rows else None
    if first_row_cells and len(first_row_cells) == table.col_count and all(c is not None for c in first_row_cells):
        weights = [max(c[2] - c[0], 1.0) for c in first_row_cells]
        try:
            hwpx_table.set_column_widths(weights)
        except Exception:
            pass

    try:
        grid = table.extract()
    except Exception:
        grid = []

    for row_index in range(table.row_count):
        row_data = grid[row_index] if row_index < len(grid) else []
        for col_index in range(table.col_count):
            cell_text = row_data[col_index] if col_index < len(row_data) else None
            if not cell_text:
                continue
            try:
                hwpx_table.set_cell_text(row_index, col_index, cell_text, split_paragraphs=True)
            except Exception:
                pass

    return hwpx_table


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

                tables = extract_page_tables(page)
                table_bboxes = [t.bbox for t in tables]
                lines = extract_page_lines(page, exclude_bboxes=table_bboxes)
                all_line_bboxes = [ln["bbox"] for ln in extract_page_lines(page)]
                images = extract_page_images(page, dpi)
                images += extract_page_vector_regions(page, dpi, text_bboxes=all_line_bboxes)

                blocks = [{"kind": "line", "bbox": ln["bbox"], "line": ln} for ln in lines]
                blocks += [{"kind": "table", "bbox": t.bbox, "table": t} for t in tables]
                blocks.sort(key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))

                page_anchor_paragraph = None
                prev_bottom_pt = 0.0
                is_first_block_on_page = True

                for block in blocks:
                    x0, y0, x1, y1 = block["bbox"]
                    gap_pt = max(0.0, y0 - prev_bottom_pt)
                    prev_bottom_pt = y1

                    if block["kind"] == "line":
                        if page_index == 0 and is_first_block_on_page:
                            paragraph = first_paragraph
                        else:
                            paragraph = doc.add_paragraph("", include_run=False)

                        for span in block["line"]["spans"]:
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
                    else:
                        hwpx_table = add_table_block(doc, block["table"])
                        paragraph = hwpx_table.paragraph

                    if page_index > 0 and is_first_block_on_page:
                        paragraph.element.set("pageBreak", "1")
                    is_first_block_on_page = False
                    if page_anchor_paragraph is None:
                        page_anchor_paragraph = paragraph

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
                        if page_index > 0 and is_first_block_on_page:
                            page_anchor_paragraph.element.set("pageBreak", "1")
                        is_first_block_on_page = False

                    for image in images:
                        ix0, iy0, ix1, iy1 = image["bbox"]
                        item_id = doc.add_image(image["png"], "png")
                        page_anchor_paragraph.add_picture(
                            item_id,
                            width=pt_to_hwpunit(ix1 - ix0),
                            height=pt_to_hwpunit(iy1 - iy0),
                            treat_as_char=False,
                            text_wrap="BEHIND_TEXT" if image.get("behind_text") else None,
                            pos_overrides={
                                "horzRelTo": "PAPER",
                                "vertRelTo": "PAPER",
                                "horzAlign": "LEFT",
                                "vertAlign": "TOP",
                                "horzOffset": pt_to_hwpunit(ix0),
                                "vertOffset": pt_to_hwpunit(iy0),
                            },
                        )

                print(f"  {page_index + 1}/{page_count} 페이지 처리 완료 "
                      f"(텍스트 줄 {len(lines)}개, 표 {len(tables)}개, 이미지 {len(images)}개)")

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
# - 회전된 텍스트는 변환되지 않습니다. 사진(래스터 이미지)은 해당 영역을 비트맵으로
#   다시 렌더링해 원래 위치에 삽입합니다. 장식용 벡터 도형(색띠, 배경 박스, 그라데이션
#   등)도 인접한 도형끼리 묶어 하나의 영역으로 비트맵 렌더링해 배치하지만, 그 영역에
#   텍스트가 겹쳐 있으면 텍스트 중복을 막기 위해 아예 건너뜁니다(그 부분은 배경색이
#   빠집니다). 페이지 전체를 덮는 큰 배경/장식은 다른 내용을 가리지 않도록 제외됩니다.
# - 표는 PyMuPDF의 표 감지 기능으로 찾아 실제 HWPX 표(<hp:tbl>)로 재구성합니다. 다만:
#     * 셀 병합(merge)은 자동으로 복원하지 않습니다 — 병합되어 있던 셀도 각각 별도
#       셀로 채워집니다.
#     * 표 안에 중첩된 작은 표가 있으면 바깥쪽 표만 만들고, 안쪽 표는 해당 셀의
#       텍스트로 평탄화됩니다(중복 삽입 방지를 위한 설계).
#     * 표 감지 자체가 실패하거나 부정확할 수 있어(특히 테두리선이 없는 표), 그런
#       경우 표가 아닌 일반 텍스트 줄로 처리됩니다.
# - 페이지 크기가 페이지마다 다른 PDF는 1페이지 크기로 통일됩니다.

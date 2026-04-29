#!/usr/bin/env python3
"""Dump every word and its bbox from a grounded-output PDF, sorted by y."""

import sys

import fitz


def main(pdf_path: str) -> None:
    doc = fitz.open(pdf_path)
    for pn, page in enumerate(doc):
        words = list(page.get_text("words"))
        # Group words by approximate baseline (y0 rounded to 5pt)
        words.sort(key=lambda w: (round(w[1] / 5) * 5, w[0]))

        prev_y = None
        line_words: list[str] = []
        line_x: tuple[float, float] | None = None
        line_y: float | None = None

        def flush():
            if line_words:
                x0, x1 = line_x
                print(f"  y={line_y:6.1f} x=({x0:6.1f},{x1:6.1f}) text={' '.join(line_words)!r}")

        print(f"page {pn}:")
        for x0, y0, x1, y1, w, *_ in words:
            row_y = round(y0 / 5) * 5
            if prev_y is None or row_y != prev_y:
                flush()
                line_words = [w]
                line_x = (x0, x1)
                line_y = y0
            else:
                line_words.append(w)
                lx0, lx1 = line_x
                line_x = (min(lx0, x0), max(lx1, x1))
            prev_y = row_y
        flush()
    doc.close()


if __name__ == "__main__":
    main(sys.argv[1])

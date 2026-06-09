"""Verify examples/*.html carry the template the current writer emits.

The OCR text and bboxes inside the example outputs come from real model
runs and can only be reproduced with a live LLM. The template parts
(stylesheet, fit-to-viewport script, span sizing units) come straight
from `pdf_ocr.core.html` constants, so any drift between the constants
and the committed examples means the examples predate a writer change
and need regenerating.

Usage:
    uv run scripts/check_examples_freshness.py [dir]

`dir` defaults to the repo's `examples/`; pass another directory to
validate freshly regenerated outputs before promoting them.

Exit code 0 when every example matches; 1 with a per-file diff report
otherwise.
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdf_ocr.core.html import _DARK_INVERT_CSS, _FIT_SCRIPT, _PAGE_CSS  # noqa: E402

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

_STYLE_RE = re.compile(r"<style>\n(.*?)</style>", re.DOTALL)


def check_file(path: Path) -> list[str]:
    """Return a list of human-readable problems for one example HTML."""
    problems: list[str] = []
    html = path.read_text(encoding="utf-8")

    expected_css = _PAGE_CSS
    if "invert_dark" in path.stem:
        expected_css += _DARK_INVERT_CSS

    m = _STYLE_RE.search(html)
    if not m:
        problems.append("no <style> block found")
    elif m.group(1) != expected_css:
        diff = "\n".join(
            difflib.unified_diff(
                expected_css.splitlines(),
                m.group(1).splitlines(),
                fromfile="current writer CSS",
                tofile=path.name,
                lineterm="",
            )
        )
        problems.append(f"stylesheet drift:\n{diff}")

    if _FIT_SCRIPT not in html:
        problems.append(
            "fit-to-viewport script differs from the writer's _FIT_SCRIPT"
        )

    # Spans must use container-relative sizing (cqw), the unit the
    # current writer emits. Pixel-based font sizes mean a pre-cqw file.
    if 'class="line"' in html and "cqw" not in html:
        problems.append("spans are not sized in cqw units (pre-cqw output)")

    return problems


def main() -> int:
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else EXAMPLES_DIR
    failures = 0
    html_files = sorted(target_dir.glob("*.html"))
    if not html_files:
        print(f"no HTML files found under {target_dir}", file=sys.stderr)
        return 1

    for path in html_files:
        problems = check_file(path)
        if problems:
            failures += 1
            print(f"STALE  {path.name}")
            for p in problems:
                print(f"       - {p}")
        else:
            print(f"OK     {path.name}")

    if failures:
        print(
            f"\n{failures} of {len(html_files)} example(s) need regenerating "
            "against the current writer.",
        )
        return 1
    print(f"\nall {len(html_files)} examples match the current writer template")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

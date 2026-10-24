"""Resolve document page selections before rasterization or inference."""


def parse_page_range(page_str: str, total_pages: int) -> list[int]:
    """Parse a 1-indexed range into sorted, unique 0-indexed source pages."""
    pages: set[int] = set()
    try:
        for part in page_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s), int(end_s)
                pages.update(range(max(1, start) - 1, min(total_pages, end)))
            else:
                page = int(part)
                if 1 <= page <= total_pages:
                    pages.add(page - 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid page range {page_str!r}; use page numbers such as '1-3,5'."
        ) from exc
    return sorted(pages)


def select_pages(pages: str | None, total_pages: int) -> list[int]:
    """Select source pages, rejecting a nonempty range that matches none."""
    if pages is None or not pages.strip():
        return list(range(total_pages))
    selected = parse_page_range(pages, total_pages)
    if not selected:
        raise ValueError(
            f"Page range {pages!r} selects no pages in a {total_pages}-page document."
        )
    return selected

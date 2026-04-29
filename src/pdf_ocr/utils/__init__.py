"""Utility modules."""

from pdf_ocr.utils.tqdm_patch import SilentTqdm, apply as apply_tqdm_patch

__all__ = ["SilentTqdm", "apply_tqdm_patch"]

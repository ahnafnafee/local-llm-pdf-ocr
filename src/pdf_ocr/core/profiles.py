"""Model input/output conventions, independent of inference transports."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ModelProfile:
    name: str = "generic"
    image_first: bool = False
    page_prompt: str | None = None
    crop_prompt: str | None = None
    grounding_prompt: str | None = None
    grounding_format: str = "json"
    temperature: float = 0.0
    grounded_temperature: float = 0.0
    grounded_retry_temperature: float | None = None
    native_layout: bool = False
    verify_inline_identifiers: bool = False
    max_image_dim: int = 1024
    sampling: tuple[tuple[str, float], ...] = ()
    grounded_sampling: tuple[tuple[str, float], ...] = ()
    grounded_max_tokens: int = 4096
    crop_canvas_size: int | None = None


_PROFILES = {
    "generic": ModelProfile(),
    "olmocr": ModelProfile(name="olmocr", temperature=0.1, crop_canvas_size=1024),
    "qwen3-vl": ModelProfile(
        name="qwen3-vl", grounded_temperature=0.7,
        grounded_sampling=(("top_p", 0.8), ("presence_penalty", 1.5)),
    ),
    "qwen3.5": ModelProfile(
        name="qwen3.5", image_first=True, grounded_temperature=0.7,
        grounded_sampling=(("top_p", 0.8), ("presence_penalty", 1.5)),
        grounded_max_tokens=8192,
    ),
    "lightonocr-3": ModelProfile(
        name="lightonocr-3", image_first=True, page_prompt="", crop_prompt="",
        grounding_prompt="grounding", grounding_format="lighton",
        temperature=0.0, grounded_temperature=0.0,
        grounded_retry_temperature=0.2, native_layout=True,
        verify_inline_identifiers=True,
        max_image_dim=2048,
        sampling=(("top_p", 0.9),), grounded_sampling=(("top_p", 0.9),),
    ),
    "paddleocr-vl": ModelProfile(
        name="paddleocr-vl", image_first=True, page_prompt="OCR:", crop_prompt="OCR:",
        grounding_prompt="Spotting:", grounding_format="paddle",
    ),
}


def resolve_profile(model: str, profile: str | None = None) -> ModelProfile:
    """Resolve known model conventions; explicit names support server aliases.

    Unknown models keep the general transcription/JSON prompts. A profile
    changes prompts and decoding conventions, never endpoint or model identity.
    """
    profile = profile if profile is not None else os.getenv("LLM_MODEL_PROFILE")
    if profile in (None, "", "auto"):
        name = model.lower()
        if "lightonocr-3" in name:
            profile = "lightonocr-3"
        elif "paddleocr-vl" in name:
            profile = "paddleocr-vl"
        elif "olmocr" in name:
            profile = "olmocr"
        elif "qwen3.5" in name or "qwen3_5" in name:
            profile = "qwen3.5"
        elif "qwen3" in name:
            profile = "qwen3-vl"
        else:
            profile = "generic"
    if profile not in _PROFILES:
        raise ValueError(f"Unknown model profile {profile!r}; choose {', '.join(_PROFILES)}")
    return _PROFILES[profile]

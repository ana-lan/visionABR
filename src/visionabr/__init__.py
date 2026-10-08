"""VisionABR: load-aware image resolution gateway for vLLM vision-language models."""

from importlib.metadata import version, PackageNotFoundError

from .sizes import (
    PIXELS_PER_TOKEN,
    SIZE_BUDGETS,
    max_pixels_for_budget,
    max_pixels_for_size,
)

try:
    __version__ = version("visionabr")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "PIXELS_PER_TOKEN",
    "SIZE_BUDGETS",
    "max_pixels_for_budget",
    "max_pixels_for_size",
]
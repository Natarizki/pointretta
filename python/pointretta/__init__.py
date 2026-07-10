"""PointRetta: retention-based LLM (O(N) training, O(1) inference)."""

from ._core import train_and_save, load_and_generate

__all__ = ["train_and_save", "load_and_generate"]
__version__ = "0.1.0"

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


DEFAULT_PERCEPTION_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_DECODER_MODEL_ID = "Qwen/Qwen3-8B"


@dataclass(frozen=True)
class TCRFormerPaths:
    """
    Placeholder artifact references for open-source release.

    Replace these values with your own local checkpoint paths or LoRA directory
    after cloning the repository.
    """

    encoder_checkpoint: Optional[Union[str, Path]] = None
    projector_checkpoint: Optional[Union[str, Path]] = None
    decoder_lora_dir: Optional[Union[str, Path]] = None


DEFAULT_PATHS = TCRFormerPaths()


def normalize_optional_path(path_like: Optional[Union[str, Path]]) -> Optional[Path]:
    """Convert an optional path-like value to ``Path`` while preserving ``None``."""
    if path_like is None:
        return None

    normalized = Path(path_like)
    return normalized if str(normalized).strip() else None

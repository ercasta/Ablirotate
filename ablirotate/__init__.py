"""
Ablirotate: Techniques for shrinking LLMs via activation-based pruning,
matrix defragmentation, and differential abliteration.
"""

from .tracker import ActivationTracker
from .pruner import ModelPruner
from .defrag import MatrixDefragmenter
from .differential import DifferentialAbliterator
from .qwen_coder import (
    QWEN_CODER_30B_CONFIG,
    QwenCoderActivationTracker,
    QwenCoderMlpPruner,
    QwenCoderDefragmenter,
    QwenCoderPipeline,
)
from .gemma4 import (
    GEMMA4_27B_CONFIG,
    Gemma4ActivationTracker,
    Gemma4MlpPruner,
    Gemma4Defragmenter,
    Gemma4Pipeline,
)

__all__ = [
    "ActivationTracker",
    "ModelPruner",
    "MatrixDefragmenter",
    "DifferentialAbliterator",
    # Qwen Coder 30B specific
    "QWEN_CODER_30B_CONFIG",
    "QwenCoderActivationTracker",
    "QwenCoderMlpPruner",
    "QwenCoderDefragmenter",
    "QwenCoderPipeline",
    # Gemma 4 specific
    "GEMMA4_27B_CONFIG",
    "Gemma4ActivationTracker",
    "Gemma4MlpPruner",
    "Gemma4Defragmenter",
    "Gemma4Pipeline",
]

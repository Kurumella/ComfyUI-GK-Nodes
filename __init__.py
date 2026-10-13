"""ComfyUI-GK-Nodes

Dataset preparation for MiniMax H3 finetuning: turn a folder of large video
clips into H3-frame-conformant clips with matching T2VA prompt .txt files.
"""

from .nodes.dataset_builder import GKH3DatasetBuilder
from .nodes.frame_calculator import GKH3FrameCalculator
from .nodes.prompt_validator import GKT2VAPromptValidator

NODE_CLASS_MAPPINGS = {
    "GKH3DatasetBuilder": GKH3DatasetBuilder,
    "GKH3FrameCalculator": GKH3FrameCalculator,
    "GKT2VAPromptValidator": GKT2VAPromptValidator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GKH3DatasetBuilder": "GK H3 Dataset Builder",
    "GKH3FrameCalculator": "GK H3 Frame Calculator",
    "GKT2VAPromptValidator": "GK T2VA Prompt Validator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

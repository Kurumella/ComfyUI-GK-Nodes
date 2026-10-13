"""GK T2VA Prompt Validator - check/normalize a prompt against T2VA structure."""

from __future__ import annotations

from ..core import t2va_format


class GKT2VAPromptValidator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t2va_text": ("STRING", {"default": "", "multiline": True}),
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "Reassemble the three fields in canonical order/spacing."}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("is_valid", "issues", "normalized_text")
    FUNCTION = "run"
    CATEGORY = "GK-Nodes/Utility"

    def run(self, t2va_text, normalize):
        result = t2va_format.validate(t2va_text)
        normalized = t2va_format.normalize(t2va_text) if normalize else t2va_format.strip_fences(t2va_text)
        return (result.ok, result.as_text(), normalized)

"""Whisper transcription for clip audio.

Produces verbatim dialogue plus per-segment timings and a detected language
code. The language is mapped to a human-readable tag used inside the T2VA
``<d>[Language] ...</d>`` dialogue blocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

LOGGER = logging.getLogger("ComfyUI.GKNodes.transcribe")

WHISPER_MODELS = [
    "tiny.en", "tiny", "base.en", "base", "small.en", "small",
    "medium.en", "medium", "large-v1", "large-v2", "large-v3",
    "large", "large-v3-turbo", "turbo",
]

_MODEL_CACHE: dict[str, object] = {}


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcription:
    text: str = ""
    language: str = ""          # ISO code, e.g. "en"
    language_name: str = ""     # e.g. "English"
    segments: list[Segment] = field(default_factory=list)

    @property
    def has_speech(self) -> bool:
        return bool(self.text.strip())


def speech_boundary_spans(trans: "Transcription | None") -> list[tuple[float, float]]:
    """Candidate (start, end) speech spans for speech-aware segmentation.

    Each Whisper segment with real text becomes one span; segment starts are
    good clip-start points and segment ends are good clip-end points. Returned
    sorted by start.
    """
    if trans is None:
        return []
    spans = [(float(s.start), float(s.end)) for s in trans.segments if s.text.strip()]
    return sorted(spans)


def slice_transcription(
    trans: "Transcription | None", start: float, end: float
) -> "Transcription | None":
    """Return the portion of `trans` overlapping [start, end], rebased to 0.

    Used to derive a clip's dialogue from a single whole-source transcription
    instead of re-running Whisper per clip. A segment is kept if it overlaps the
    window at all; its timestamps are shifted so the clip starts at 0.
    """
    if trans is None:
        return None
    segs: list[Segment] = []
    for s in trans.segments:
        if s.end <= start or s.start >= end:
            continue
        segs.append(
            Segment(
                start=max(0.0, s.start - start),
                end=min(end, s.end) - start,
                text=s.text,
            )
        )
    text = " ".join(s.text.strip() for s in segs).strip()
    return Transcription(
        text=text,
        language=trans.language,
        language_name=trans.language_name,
        segments=segs,
    )


def _whisper_download_root() -> str | None:
    try:
        import folder_paths
    except Exception:
        return None
    import os
    for key in ("stt_whisper", "whisper"):
        try:
            paths = folder_paths.get_folder_paths(key)
            if paths:
                return paths[0]
        except Exception:
            pass
    try:
        paths = folder_paths.get_folder_paths("stt")
        if paths:
            return os.path.join(paths[0], "whisper")
    except Exception:
        pass
    return os.path.join(folder_paths.models_dir, "stt", "whisper")


def _load_model(name: str):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    import whisper  # openai-whisper
    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
        device = "cuda" if str(device).startswith("cuda") else "cpu"
    except Exception:
        device = "cpu"
    download_root = _whisper_download_root()
    LOGGER.info("Loading Whisper model '%s' on %s", name, device)
    model = whisper.load_model(name, download_root=download_root, device=device)
    _MODEL_CACHE[name] = model
    return model


def _language_name(code: str) -> str:
    try:
        import whisper
        return whisper.tokenizer.LANGUAGES.get((code or "").lower(), code or "").title() or code
    except Exception:
        return code


def _language_code(name: str) -> str | None:
    if not name or name.lower() == "auto":
        return None
    try:
        import whisper
        by_name = {v.lower(): k for k, v in whisper.tokenizer.LANGUAGES.items()}
        return by_name.get(name.lower(), name.lower())
    except Exception:
        return name.lower()


def transcribe_media(
    media_path: str,
    *,
    model_name: str = "large-v3",
    language: str = "auto",
) -> Transcription:
    """Transcribe a media file (video or audio). Returns an empty Transcription
    on any failure or when there is no detectable speech."""
    try:
        model = _load_model(model_name)
    except Exception as error:
        LOGGER.warning("Whisper model '%s' failed to load: %s", model_name, error)
        return Transcription()

    kwargs: dict[str, object] = {"word_timestamps": True}
    code = _language_code(language)
    if code:
        kwargs["language"] = code

    try:
        result = model.transcribe(str(media_path), **kwargs)
    except Exception as error:
        LOGGER.warning("Transcription failed for %s: %s", media_path, error)
        return Transcription()

    detected = str(result.get("language") or code or "")
    segments = [
        Segment(start=float(s.get("start", 0.0)), end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")).strip())
        for s in result.get("segments", [])
        if str(s.get("text", "")).strip()
    ]
    return Transcription(
        text=str(result.get("text", "")).strip(),
        language=detected,
        language_name=_language_name(detected),
        segments=segments,
    )

"""Caption a clip into MiniMax H3 T2VA format using a local Qwen3.8 VLM.

This reuses the already-installed llama.cpp runtime and GGUF models from the
sibling ``ComfyUI_Qwen_H3_Prompt`` package (so the heavy per-machine runtime is
not duplicated), but builds its own T2VA-focused prompt from the bundled
``references/t2va_guide_base_en.txt`` guide and samples frames straight from the
clip file. If the sibling package or its runtime is unavailable, captioning is
skipped gracefully and the caller falls back to a template-only prompt.
"""

from __future__ import annotations

import base64
import importlib
import io
import logging
import math
from pathlib import Path

LOGGER = logging.getLogger("ComfyUI.GKNodes.caption")

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
GUIDE_PATH = PACKAGE_ROOT / "references" / "t2va_guide_base_en.txt"
SIBLING_PACKAGE = "ComfyUI_Qwen_H3_Prompt"


class CaptionUnavailable(RuntimeError):
    """Raised when the local VLM runtime cannot be used."""


# --------------------------------------------------------------------------- #
# Sibling runtime (llama.cpp) access
# --------------------------------------------------------------------------- #
def _import_sibling(submodule: str):
    """Import a submodule of the sibling Qwen H3 package robustly."""
    name = f"{SIBLING_PACKAGE}.{submodule}"
    try:
        return importlib.import_module(name)
    except Exception:
        pass
    # Fall back to loading directly from the sibling directory on disk.
    import importlib.util
    import sys
    sibling_dir = PACKAGE_ROOT.parent / SIBLING_PACKAGE
    init = sibling_dir / "__init__.py"
    file = sibling_dir / f"{submodule}.py"
    if not file.is_file():
        raise CaptionUnavailable(
            f"Sibling package '{SIBLING_PACKAGE}' not found next to ComfyUI-GK-Nodes. "
            "Install ComfyUI_Qwen_H3_Prompt (and run its install_runtime.py) to enable captioning."
        )
    if str(sibling_dir.parent) not in sys.path:
        sys.path.insert(0, str(sibling_dir.parent))
    if init.is_file() and SIBLING_PACKAGE not in sys.modules:
        try:
            importlib.import_module(SIBLING_PACKAGE)
        except Exception:
            pass
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _models_dir() -> Path:
    import folder_paths
    return Path(folder_paths.models_dir) / "LLM" / "Qwen3.8"


def resolve_models(llm_model: str = "auto", vision_model: str = "auto") -> tuple[Path, Path]:
    root = _models_dir()
    if not root.is_dir():
        raise CaptionUnavailable(f"Model directory not found: {root}")
    gguf = sorted(root.glob("*.gguf"))
    projectors = [p for p in gguf if "mmproj" in p.name.lower()]
    weights = [p for p in gguf if "mmproj" not in p.name.lower()]
    if not weights:
        raise CaptionUnavailable(f"No Qwen GGUF weights found in {root}")
    if not projectors:
        raise CaptionUnavailable(f"No mmproj (vision) GGUF found in {root}")

    def pick(name: str, options: list[Path]) -> Path:
        if name and name != "auto":
            match = next((p for p in options if p.name == name), None)
            if match:
                return match
        return options[0]

    return pick(llm_model, weights), pick(vision_model, projectors)


def model_options() -> tuple[list[str], list[str]]:
    """(weights, projectors) filenames for node combo widgets; 'auto' first."""
    root = _models_dir()
    gguf = sorted(p.name for p in root.glob("*.gguf")) if root.is_dir() else []
    projectors = [n for n in gguf if "mmproj" in n.lower()]
    weights = [n for n in gguf if "mmproj" not in n.lower()]
    return (["auto", *weights], ["auto", *projectors])


# Default captioning GGUFs to fetch when auto_download_models is enabled and the
# model directory is empty. (repo_id, filename, approx size for the log). The
# abliterated 12 GB weight fits 16 GB-VRAM cards; mmproj comes from the unsloth
# GGUF repo. Both are single public files (verified against the Hub).
DEFAULT_MMPROJ = ("unsloth/Qwen3.8-27B-GGUF", "mmproj-F16.gguf", "~0.9 GB")
DEFAULT_WEIGHTS = (
    "soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf",
    "qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf",
    "~12.6 GB",
)


def models_present() -> bool:
    """True when at least one weights GGUF and one mmproj GGUF are already there."""
    root = _models_dir()
    if not root.is_dir():
        return False
    gguf = list(root.glob("*.gguf"))
    weights = [p for p in gguf if "mmproj" not in p.name.lower()]
    projectors = [p for p in gguf if "mmproj" in p.name.lower()]
    return bool(weights and projectors)


def download_default_models(progress=None) -> Path:
    """Download the default Qwen3.8 captioning GGUFs into models/LLM/Qwen3.8/.

    Opt-in (the caller gates this on a toggle) because the weights are ~12.6 GB.
    Returns the model directory; raises CaptionUnavailable on any failure so the
    caller can fall back to template prompts. `progress` is an optional callable
    taking one string, used to log each file.
    """
    def _say(msg: str) -> None:
        if progress:
            progress(msg)

    try:
        from huggingface_hub import hf_hub_download
    except Exception as error:  # pragma: no cover - env without huggingface_hub
        raise CaptionUnavailable(f"huggingface_hub is not available: {error}")

    root = _models_dir()
    root.mkdir(parents=True, exist_ok=True)
    for repo_id, filename, approx in (DEFAULT_MMPROJ, DEFAULT_WEIGHTS):
        target = root / filename
        if target.is_file():
            _say(f"  already present: {filename}")
            continue
        _say(f"  downloading {filename} ({approx}) from {repo_id} ...")
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(root))
        except Exception as error:
            raise CaptionUnavailable(f"failed to download {filename} from {repo_id}: {error}")
        _say(f"  downloaded {filename}")
    return root


# --------------------------------------------------------------------------- #
# Frame sampling -> JPEG data URLs (contact sheets)
# --------------------------------------------------------------------------- #
def _read_frames(clip_path: str, max_frames: int) -> list["object"]:
    """Read up to max_frames evenly-spaced RGB frames (PIL Images) from a clip."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[object] = []
    if total <= 0:
        # Unknown length: read sequentially up to max_frames.
        while len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()
        return frames

    count = min(max_frames, total)
    indices = sorted({int(round(i * (total - 1) / max(1, count - 1))) for i in range(count)})
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


def _contact_sheet_data_url(frames: list, max_edge: int = 1280) -> str:
    from PIL import Image

    if not frames:
        raise CaptionUnavailable("No frames could be read from the clip.")
    columns = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / columns)
    cell = max(1, max_edge // max(columns, rows))
    canvas = Image.new("RGB", (columns * cell, rows * cell), (16, 16, 16))
    for pos, frame in enumerate(frames):
        thumb = frame.copy()
        thumb.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        col, row = pos % columns, pos // columns
        x = col * cell + (cell - thumb.width) // 2
        y = row * cell + (cell - thumb.height) // 2
        canvas.paste(thumb, (x, y))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def _load_guide() -> str:
    try:
        return GUIDE_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def _system_prompt() -> str:
    guide = _load_guide()
    return (
        "You are a MiniMax H3 video-prompt writer. You will be shown sampled frames "
        "from ONE short video clip and must describe it as a Text-to-Video-with-Audio "
        "(T2VA) prompt that could regenerate the clip.\n\n"
        "Follow this official guide exactly:\n\n"
        f"{guide}\n\n"
        "OUTPUT RULES:\n"
        "- Produce ONLY the three T2VA core fields, in this exact order and with these "
        "exact labels, each separated by one blank line:\n"
        "  integrated_multimodal_description: ...\n"
        "  overall_soundscape: ...\n"
        "  non_diegetic_music: ...\n"
        "- This is a T2VA task: do NOT write any first/last-frame alignment instruction line.\n"
        "- Begin integrated_multimodal_description with [Shot 1] and the visual style.\n"
        "- Use increasing cut timestamps for later shots, within the clip duration.\n"
        "- If a transcript is provided, reproduce the spoken words VERBATIM inside "
        "<d>[Language] ...</d> blocks on the speaking subject; never translate or invent dialogue.\n"
        "- Keep one continuous spoken passage in a SINGLE <d> block. Do NOT break a "
        "sentence or a continuous line across multiple <d> blocks. Describe facial "
        "expressions, gestures and camera motion before or after the <d> block, not by "
        "splitting the speech.\n"
        "- Split a line into more than one <d> block ONLY when it crosses a shot cut "
        "(put <scenetrans> in both parts and state the audio continues across the cut) "
        "or is truncated by the end of the clip (end the final part with <cutoff>).\n"
        "- End every completed <d> block with terminal punctuation (., ? or !). A <d> "
        "block may end without terminal punctuation only when it is immediately marked "
        "with <cutoff> (speech cut off by the clip end) or <scenetrans> (continues across a cut).\n"
        "- If no transcript is provided, do not invent spoken dialogue.\n"
        "- Do not add commentary, markdown fences, headings, or any text outside the three fields."
    )


def _transcript_block(transcription) -> str:
    if transcription is None or not getattr(transcription, "has_speech", False):
        return "Transcript: (none - the clip has no intelligible speech; do not invent dialogue)."
    lang = transcription.language_name or transcription.language or "unknown"
    # The full verbatim text is what must be reproduced -- as ONE <d> block for a
    # continuous passage. Per-segment timings are given only for timeline
    # placement; the model must NOT split a continuous line on segment boundaries.
    full = " ".join(seg.text.strip() for seg in transcription.segments).strip()
    if not full:
        full = (transcription.text or "").strip()
    lines = [
        f"Transcript language: {lang}",
        "Reproduce this spoken passage VERBATIM, as a SINGLE <d> block on the speaking "
        "subject (do not split it across multiple <d> blocks):",
        f"  {full}",
    ]
    if transcription.segments:
        lines.append(
            "Segment timings (for placing the speech along the timeline only - do NOT "
            "use these as <d> split points):"
        )
        for seg in transcription.segments:
            lines.append(f"    [{seg.start:.2f}-{seg.end:.2f}s] {seg.text.strip()}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
class CaptionEngine:
    """Holds a resident llama-server across many clips; release() when done."""

    def __init__(self, llm_model: str = "auto", vision_model: str = "auto",
                 context_size: int = 32768):
        self._runtime = _import_sibling("runtime")
        self._manager = self._runtime.LlamaServerManager()
        self._llm_path, self._mmproj_path = resolve_models(llm_model, vision_model)
        self._context_size = context_size
        self._server = None

    @property
    def model_name(self) -> str:
        return self._llm_path.name

    def _ensure_server(self):
        if self._server is None or not self._server.is_running:
            try:
                import comfy.model_management as mm
                mm.unload_all_models()
                mm.soft_empty_cache()
            except Exception:
                pass
            self._server, _ = self._manager.acquire(
                self._llm_path, self._mmproj_path, self._context_size
            )
        return self._server

    def caption_clip(
        self,
        clip_path: str,
        duration: float,
        *,
        transcription=None,
        style_hint: str = "",
        extra_instructions: str = "",
        max_sample_frames: int = 12,
        seed: int = 0,
        max_tokens: int = 4096,
    ) -> str:
        server = self._ensure_server()

        frames = _read_frames(clip_path, max_sample_frames)
        sheet = _contact_sheet_data_url(frames)

        request_lines = [
            f"Target clip duration: {duration:.2f} seconds (constant 24 fps).",
            "The image is a chronological contact sheet of frames sampled evenly across "
            "the clip; read cells in row-major (left-to-right, top-to-bottom) time order.",
            _transcript_block(transcription),
        ]
        if style_hint.strip():
            request_lines.append(f"Style / content guidance from the user: {style_hint.strip()}")
        if extra_instructions.strip():
            request_lines.append(extra_instructions.strip())

        content = [
            {"type": "text", "text": "\n\n".join(request_lines)},
            {"type": "image_url", "image_url": {"url": sheet, "detail": "auto"}},
        ]
        messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": content},
        ]

        answer, _usage = server.chat(
            messages,
            seed=seed,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            think_mode=False,
            reasoning_effort="low",
        )
        return answer.strip()

    def release(self) -> None:
        try:
            self._manager.release()
        except Exception:
            pass
        self._server = None

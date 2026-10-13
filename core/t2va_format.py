"""Build, normalize and validate MiniMax H3 T2VA prompt text.

T2VA (Text-to-Video-with-Audio) prompts have exactly three core fields in a
fixed order, with no first/last-frame alignment instruction line:

    integrated_multimodal_description: [Shot 1] ...
    overall_soundscape: ...
    non_diegetic_music: ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")

_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n```$", re.MULTILINE)


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text).strip()
    return text


def _extract_field(text: str, field: str, next_fields: tuple[str, ...]) -> str | None:
    """Pull the value of one labelled field out of a free-form block.

    The value runs from the field's colon up to the next field label, or to the
    end of the text when `next_fields` is empty (so a colon inside the final
    field's value does not truncate it).
    """
    if next_fields:
        stop = "(?=" + "|".join(rf"{re.escape(f)}\s*:" for f in next_fields) + r"|\Z)"
    else:
        stop = r"(?=\Z)"
    pattern = rf"{re.escape(field)}\s*:\s*(.*?){stop}"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def normalize(text: str) -> str:
    """Reassemble the three fields in canonical order and spacing.

    If a field is missing it is emitted with an ``N/A`` placeholder so the file
    is always structurally complete; the validator will still report the gap.
    """
    text = strip_fences(text)
    values: dict[str, str] = {}
    for i, field in enumerate(FIELDS):
        value = _extract_field(text, field, FIELDS[i + 1:])
        values[field] = (value or "N/A").strip()

    block = []
    for field in FIELDS:
        block.append(f"{field}: {values[field]}")
    return "\n\n".join(block).strip() + "\n"


def apply_trigger_word(text: str, trigger_word: str) -> str:
    """Insert an optional LoRA trigger word into the visual description.

    A trigger word is a unique activation token shared by every prompt in a
    finetuning dataset, so training binds the token to the dataset's
    subject/style and naming it at inference time activates the learned
    concept. It is placed at the start of the clip's described content - right
    after the mandatory ``[Shot 1]`` marker of
    ``integrated_multimodal_description`` - so it reads as the first content
    token while keeping the T2VA structure intact.

    No-op for an empty trigger word, and idempotent: a trigger already present
    at that position is not duplicated.
    """
    trigger = (trigger_word or "").strip()
    if not trigger:
        return text

    def _inject(match: "re.Match[str]") -> str:
        label, body = match.group(1), match.group(2)
        shot = re.match(r"(\s*\[Shot 1\]\s*)(.*)", body, flags=re.IGNORECASE | re.DOTALL)
        head, rest = (shot.group(1), shot.group(2)) if shot else ("", body)
        if rest.lstrip().lower().startswith(trigger.lower()):
            return match.group(0)
        joined = f"{trigger}, {rest}" if rest.strip() else trigger
        return f"{label}{head}{joined}"

    pattern = re.compile(
        r"(integrated_multimodal_description\s*:[ \t]*)(.*?)"
        r"(?=\n\s*overall_soundscape\s*:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return pattern.sub(_inject, text, count=1)


@dataclass
class ValidationResult:
    ok: bool
    issues: list[str]

    def as_text(self) -> str:
        if self.ok and not self.issues:
            return "VALID: prompt conforms to T2VA structure."
        prefix = "VALID (with warnings)" if self.ok else "INVALID"
        return prefix + ":\n" + "\n".join(f" - {m}" for m in self.issues)


def validate(text: str) -> ValidationResult:
    issues: list[str] = []
    stripped = strip_fences(text)

    positions = {}
    for field in FIELDS:
        m = re.search(rf"\b{re.escape(field)}\s*:", stripped, flags=re.IGNORECASE)
        if m is None:
            issues.append(f"missing required field '{field}'")
        else:
            positions[field] = m.start()

    present = [f for f in FIELDS if f in positions]
    ordered = sorted(present, key=lambda f: positions[f])
    if ordered != present:
        issues.append("fields are out of order; expected: " + " -> ".join(FIELDS))

    if "integrated_multimodal_description" in positions:
        body = _extract_field(stripped, "integrated_multimodal_description",
                              ("overall_soundscape", "non_diegetic_music")) or ""
        if "[Shot 1]" not in body:
            issues.append("integrated_multimodal_description must begin with '[Shot 1]'")
        # Dialogue tags must be balanced.
        if body.count("<d>") != body.count("</d>"):
            issues.append("unbalanced <d>...</d> dialogue tags")
        # Each completed <d> block must end with terminal punctuation. A block may
        # end unpunctuated only when it is truncated by the clip end (<cutoff>) or
        # continues across a cut (<scenetrans>) -- otherwise a continuous line was
        # wrongly split mid-sentence into multiple <d> blocks.
        for m in re.finditer(r"<d>(.*?)</d>", body, flags=re.DOTALL):
            inner = re.sub(r"^\s*\[[^\]]+\]\s*", "", m.group(1))  # drop [Language] tag
            inner = inner.rstrip(" \t\n\"'”’)]}»")               # ignore trailing quotes/brackets
            after = body[m.end():m.end() + 40]
            before = body[max(0, m.start() - 40):m.start()]
            marked = "<cutoff>" in after or "<scenetrans>" in before + after
            if inner and inner[-1] not in ".?!…" and not marked:
                snippet = inner[-32:].strip()
                issues.append(
                    f"a <d> block ends without terminal punctuation and no "
                    f"<cutoff>/<scenetrans> marker (…“{snippet}”); a continuous "
                    "line split across <d> blocks should be a single block"
                )
        # Later shot cut timestamps should be present if multiple shots exist.
        shots = re.findall(r"\[Shot (\d+)\]", body)
        if len(shots) > 1:
            # [Shot 1] has no timestamp; shots >= 2 should carry one.
            tail = body.split("[Shot 2]", 1)[-1] if "[Shot 2]" in body else ""
            if "[Shot 2]" in body and not re.search(r"\d{1,2}:\d{2}\.\d{2}|\d+(?:\.\d+)?\s*seconds?", tail):
                issues.append("multi-shot description should include increasing cut timestamps")

    ok = not any(m.startswith("missing") or "out of order" in m or "begin with" in m for m in issues)
    return ValidationResult(ok=ok, issues=issues)


def build_template_prompt(
    duration: float,
    *,
    transcription=None,
    style_hint: str = "",
) -> str:
    """Deterministic T2VA fallback when the VLM is unavailable.

    Uses only what we can know without a vision model: duration, user style
    hint and the verbatim transcript. Visual detail is intentionally minimal.
    """
    style = style_hint.strip() or "Live-action, cinematic"
    desc = (
        f"[Shot 1] {style}. A single continuous {duration:.2f}-second take; "
        "the camera holds a mostly static shot on the main subject as the scene unfolds."
    )
    if transcription is not None and getattr(transcription, "has_speech", False):
        lang = transcription.language_name or transcription.language or "English"
        spoken = " ".join(seg.text for seg in transcription.segments) or transcription.text
        spoken = spoken.strip()
        if spoken:
            desc += (
                f" The on-screen subject (S1) speaks: <d>[{lang}] {spoken}</d>"
            )
        soundscape = "Light room ambience underlies the scene with subtle movement sounds."
    else:
        soundscape = "Quiet ambient room tone with faint environmental sound throughout."

    fields = [
        f"integrated_multimodal_description: {desc}",
        f"overall_soundscape: {soundscape}",
        "non_diegetic_music: N/A",
    ]
    return "\n\n".join(fields).strip() + "\n"

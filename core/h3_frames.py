"""MiniMax H3 frame mathematics.

Everything the dataset builder produces must satisfy the H3 block formula:

    total_frames = 17 * n + 5          (n = number of blocks)

At the reference rate of 24 fps the usable range is n in [7, 21]:

    n = 7  -> 124 frames -> ~5.17 s   (minimum)
    n = 21 -> 362 frames -> ~15.08 s  (maximum)

These helpers convert freely between n, frame counts and seconds, snap an
arbitrary duration down to the nearest valid clip, and greedily split a long
span (e.g. a detected scene) into back-to-back H3-valid clips.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_N = 7
MAX_N = 21
H3_FPS = 24.0


def frames_for_n(n: int) -> int:
    """Exact total frames for a given block count: 17*n + 5."""
    return (17 * n) + 5


def duration_for_n(n: int, fps: float = H3_FPS) -> float:
    """Clip duration in seconds for a block count at the given fps."""
    return frames_for_n(n) / fps


def n_for_frames(frames: int) -> int | None:
    """Return n if `frames` is an exact 17*n+5 value, else None."""
    if frames < frames_for_n(MIN_N):
        return None
    remainder = frames - 5
    if remainder % 17 != 0:
        return None
    return remainder // 17


def valid_frame_counts() -> list[int]:
    """Every valid total-frame value for n in [MIN_N, MAX_N]."""
    return [frames_for_n(n) for n in range(MIN_N, MAX_N + 1)]


def is_valid_frame_count(frames: int) -> bool:
    n = n_for_frames(frames)
    return n is not None and MIN_N <= n <= MAX_N


def clamp_n(n: int) -> int:
    return max(MIN_N, min(MAX_N, int(n)))


def best_n_for_duration(
    seconds: float,
    *,
    min_n: int = MIN_N,
    max_n: int = MAX_N,
    fps: float = H3_FPS,
) -> int | None:
    """Largest block count whose clip fits within `seconds`.

    Returns None when `seconds` is shorter than the shortest allowed clip,
    i.e. the span is too short to yield any H3-valid clip.
    """
    lo = clamp_n(min_n)
    hi = clamp_n(max_n)
    if hi < lo:
        lo, hi = hi, lo
    best: int | None = None
    for n in range(lo, hi + 1):
        if duration_for_n(n, fps) <= seconds + 1e-6:
            best = n
        else:
            break
    return best


@dataclass(frozen=True)
class ClipSpec:
    """One H3-valid clip to be cut from a source span."""

    start: float            # seconds into the source video
    n: int                  # block count
    frames: int             # 17*n + 5
    duration: float         # frames / fps

    @property
    def end(self) -> float:
        return self.start + self.duration


def split_span_into_clips(
    span_start: float,
    span_duration: float,
    *,
    min_n: int = MIN_N,
    max_n: int = MAX_N,
    fps: float = H3_FPS,
) -> list[ClipSpec]:
    """Greedily tile a span with back-to-back H3-valid clips.

    Takes the longest allowed clip that still fits in the remaining span,
    advances, and repeats. Any tail shorter than the minimum clip is dropped.
    Clips never extend past `span_start + span_duration`, so when this is called
    per detected scene the clips never cross a scene cut.
    """
    clips: list[ClipSpec] = []
    cursor = span_start
    remaining = span_duration
    while True:
        n = best_n_for_duration(remaining, min_n=min_n, max_n=max_n, fps=fps)
        if n is None:
            break
        frames = frames_for_n(n)
        duration = frames / fps
        clips.append(ClipSpec(start=cursor, n=n, frames=frames, duration=duration))
        cursor += duration
        remaining -= duration
    return clips


def describe_limits(fps: float = H3_FPS) -> str:
    lo, hi = frames_for_n(MIN_N), frames_for_n(MAX_N)
    return (
        f"H3 clips @ {fps:g}fps: n={MIN_N}->{lo} frames (~{lo / fps:.2f}s) .. "
        f"n={MAX_N}->{hi} frames (~{hi / fps:.2f}s)"
    )

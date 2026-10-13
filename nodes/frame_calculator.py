"""GK H3 Frame Calculator - utility node for the 17*n+5 relationship."""

from __future__ import annotations

from ..core.h3_frames import (
    H3_FPS,
    MAX_N,
    MIN_N,
    best_n_for_duration,
    duration_for_n,
    frames_for_n,
    is_valid_frame_count,
    n_for_frames,
    split_span_into_clips,
)


class GKH3FrameCalculator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["frames_from_n", "n_from_frames", "fit_duration", "tile_duration"], {"default": "frames_from_n"}),
                "n": ("INT", {"default": 7, "min": 1, "max": 100, "step": 1}),
                "frames": ("INT", {"default": 124, "min": 1, "max": 100000, "step": 1}),
                "duration": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": H3_FPS, "min": 1.0, "max": 240.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("frames", "n", "duration_sec", "valid", "info")
    FUNCTION = "calc"
    CATEGORY = "GK-Nodes/Utility"

    def calc(self, mode, n, frames, duration, fps):
        if mode == "frames_from_n":
            out_frames = frames_for_n(n)
            out_n = n
            out_dur = out_frames / fps
            valid = MIN_N <= n <= MAX_N
            info = f"n={n} -> {out_frames} frames (~{out_dur:.3f}s @ {fps:g}fps)" + \
                   ("" if valid else f" [outside H3 range n={MIN_N}..{MAX_N}]")

        elif mode == "n_from_frames":
            resolved = n_for_frames(frames)
            out_frames = frames
            out_n = resolved if resolved is not None else -1
            out_dur = frames / fps
            valid = is_valid_frame_count(frames)
            if resolved is None:
                info = f"{frames} frames is NOT of the form 17*n+5"
            else:
                info = f"{frames} frames -> n={resolved} (~{out_dur:.3f}s)" + \
                       ("" if valid else f" [n outside {MIN_N}..{MAX_N}]")

        elif mode == "fit_duration":
            resolved = best_n_for_duration(duration, fps=fps)
            if resolved is None:
                out_n, out_frames, out_dur, valid = -1, 0, 0.0, False
                info = f"{duration:.3f}s is shorter than the minimum H3 clip (~{duration_for_n(MIN_N, fps):.3f}s)"
            else:
                out_n = resolved
                out_frames = frames_for_n(resolved)
                out_dur = out_frames / fps
                valid = True
                info = f"{duration:.3f}s fits n={resolved} -> {out_frames} frames (~{out_dur:.3f}s)"

        else:  # tile_duration
            clips = split_span_into_clips(0.0, duration, fps=fps)
            out_n = len(clips)
            out_frames = sum(c.frames for c in clips)
            out_dur = sum(c.duration for c in clips)
            valid = bool(clips)
            parts = ", ".join(f"n={c.n}({c.frames}f)" for c in clips) or "none"
            info = (f"{duration:.3f}s tiles into {len(clips)} clip(s): {parts}; "
                    f"covered {out_dur:.3f}s, dropped {max(0.0, duration - out_dur):.3f}s tail")

        return (int(out_frames), int(out_n), float(out_dur), bool(valid), info)

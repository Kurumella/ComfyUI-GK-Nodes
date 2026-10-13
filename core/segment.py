"""Segment a source video into H3-valid clips.

Three strategies:
  * ``scene_detect`` - PySceneDetect finds natural cuts, then each scene is
    greedily tiled with H3-valid clips (clips never cross a cut). Falls back to
    ``fixed_window`` if PySceneDetect is not installed.
  * ``fixed_window``  - the whole video is tiled with back-to-back H3 clips.
  * ``single_clip``   - one H3 clip from the start of the video.
"""

from __future__ import annotations

import logging

from .ffmpeg_utils import VideoInfo
from .h3_frames import (
    ClipSpec,
    best_n_for_duration,
    duration_for_n,
    frames_for_n,
    split_span_into_clips,
)

LOGGER = logging.getLogger("ComfyUI.GKNodes.segment")


def scenedetect_available() -> bool:
    try:
        import scenedetect  # noqa: F401
        return True
    except Exception:
        return False


def _detect_scenes(path: str, threshold: float, min_scene_len_frames: int, fps: float) -> list[tuple[float, float]]:
    """Return (start_sec, end_sec) scene spans using PySceneDetect content detection."""
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(path)
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=max(1, int(min_scene_len_frames)))
    )
    manager.detect_scenes(video, show_progress=False)
    scenes = manager.get_scene_list()
    spans: list[tuple[float, float]] = []
    for start, end in scenes:
        spans.append((start.get_seconds(), end.get_seconds()))
    return spans


def _speech_aware_clips(
    duration: float,
    spans: list[tuple[float, float]],
    *,
    min_n: int,
    max_n: int,
    fps: float,
    lead_in: float = 0.3,
    tol_steps: float = 1.0,
) -> list[ClipSpec]:
    """Greedily tile dialogue so clip boundaries fall near speech boundaries.

    H3 durations are quantized (``17n+5`` frames -> 15 discrete lengths), so a
    clip's length cannot match a sentence exactly. Each clip therefore starts at
    a speech boundary and we pick the ``n`` whose END lands closest to a speech
    boundary (biasing toward larger ``n`` for more content). When no ``n`` ends
    within tolerance of a boundary, we take the largest fitting ``n`` and
    continue seamlessly rather than drop content. Mid-sentence cuts are
    minimized, not eliminated -- that is the best quantization allows.
    """
    if not spans:
        return []
    step = (frames_for_n(min_n + 1) - frames_for_n(min_n)) / fps  # 17/fps
    ends = sorted(e for _, e in spans)
    starts = sorted(s for s, _ in spans)
    tol = tol_steps * step

    clips: list[ClipSpec] = []
    cursor = max(0.0, starts[0] - lead_in)
    while True:
        remaining = duration - cursor
        n_fit = best_n_for_duration(remaining, min_n=min_n, max_n=max_n, fps=fps)
        if n_fit is None:
            break
        best_n = n_fit
        best_dist = float("inf")
        for n in range(min_n, n_fit + 1):
            end = cursor + duration_for_n(n, fps)
            dist = min(abs(end - e) for e in ends)
            # Strictly closer wins; equal distance prefers the larger n (more
            # content, fewer clips) since we iterate n ascending.
            if dist < best_dist - 1e-6:
                best_dist = dist
                best_n = n
        n = best_n if best_dist <= tol else n_fit
        frames = frames_for_n(n)
        clip_dur = frames / fps
        clips.append(ClipSpec(start=cursor, n=n, frames=frames, duration=clip_dur))
        end = cursor + clip_dur
        if best_dist <= tol:
            # Aligned on a boundary: start the next clip at the following speech
            # start, dropping inter-sentence silence. The ``end - tol`` floor lets
            # a small overshoot still pick up the next sentence instead of
            # skipping it (a strict ``> end`` would drop a whole sentence when the
            # clip ends just past its start).
            later = [s for s in starts if s > clips[-1].start + 1e-3 and s >= end - tol]
            cursor = later[0] if later else end
        else:
            # Forced cut mid-speech: continue seamlessly, no gap.
            cursor = end
        if cursor <= clips[-1].start + 1e-6:  # guarantee forward progress
            cursor = end
    return clips


def plan_clips(
    info: VideoInfo,
    *,
    strategy: str = "scene_detect",
    min_n: int = 7,
    max_n: int = 21,
    fps: float = 24.0,
    scene_threshold: float = 27.0,
    speech_spans: list[tuple[float, float]] | None = None,
) -> list[ClipSpec]:
    """Produce the ordered list of H3 clips to cut from one source video."""
    duration = info.duration
    if duration <= 0:
        return []

    if strategy == "single_clip":
        n = best_n_for_duration(duration, min_n=min_n, max_n=max_n, fps=fps)
        if n is None:
            return []
        frames = frames_for_n(n)
        return [ClipSpec(start=0.0, n=n, frames=frames, duration=frames / fps)]

    if strategy == "speech_aware":
        if speech_spans:
            clips = _speech_aware_clips(
                duration, speech_spans, min_n=min_n, max_n=max_n, fps=fps
            )
            if clips:
                return clips
            LOGGER.info(
                "Speech-aware segmentation produced no clips for %s; using fixed_window.",
                info.path.name,
            )
        else:
            LOGGER.info(
                "No speech detected in %s; using fixed_window segmentation.",
                info.path.name,
            )
        return split_span_into_clips(0.0, duration, min_n=min_n, max_n=max_n, fps=fps)

    if strategy == "scene_detect":
        if not scenedetect_available():
            LOGGER.warning(
                "PySceneDetect is not installed; falling back to fixed_window segmentation. "
                "Install with: pip install scenedetect[opencv]"
            )
        else:
            try:
                # Require detected scenes to be at least one minimum clip long.
                # PySceneDetect measures ``min_scene_len`` in SOURCE frames, so it
                # must be derived from the source fps (``info.fps``), not the H3
                # target fps -- otherwise a high-fps source (e.g. 60fps phone
                # footage) yields scenes that pass detection yet are too short in
                # real seconds to hold any H3 clip, dropping every clip.
                min_clip_seconds = duration_for_n(min_n, fps)
                source_fps = info.fps if info.fps and info.fps > 0 else fps
                min_len_frames = max(1, round(min_clip_seconds * source_fps))
                spans = _detect_scenes(str(info.path), scene_threshold, min_len_frames, fps)
                if not spans:
                    # PySceneDetect returns an EMPTY list when it finds no cuts
                    # (a single continuous shot, e.g. one long phone take). That
                    # is not a failure -- treat the whole video as one scene so
                    # it still tiles into H3 clips under scene_detect.
                    LOGGER.info(
                        "No scene cuts detected in %s; treating the whole video as one scene.",
                        info.path.name,
                    )
                    spans = [(0.0, duration)]
                clips: list[ClipSpec] = []
                for span_start, span_end in spans:
                    clips.extend(
                        split_span_into_clips(
                            span_start,
                            max(0.0, span_end - span_start),
                            min_n=min_n,
                            max_n=max_n,
                            fps=fps,
                        )
                    )
                if clips:
                    return clips
                LOGGER.warning(
                    "Scene detection produced no H3-valid clips for %s; falling back to fixed_window.",
                    info.path.name,
                )
            except Exception as error:  # pragma: no cover - defensive
                LOGGER.warning(
                    "Scene detection failed for %s (%s); falling back to fixed_window.",
                    info.path.name, error,
                )

    # fixed_window (and all fallbacks): tile the whole duration.
    return split_span_into_clips(0.0, duration, min_n=min_n, max_n=max_n, fps=fps)

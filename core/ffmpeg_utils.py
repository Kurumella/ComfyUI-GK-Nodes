"""ffmpeg/ffprobe helpers for probing sources and cutting exact-length clips."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg", ".ts"}


class FFmpegError(RuntimeError):
    pass


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegError(
            f"'{name}' was not found on PATH. Install ffmpeg (which bundles {name}) "
            "and make sure it is available to the ComfyUI process."
        )
    return path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    fps: float
    nb_frames: int
    width: int
    height: int
    has_audio: bool


def _parse_fraction(value: str) -> float:
    value = (value or "").strip()
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(path: str | Path) -> VideoInfo:
    """Probe a media file for duration, fps, frame count, size and audio."""
    path = Path(path)
    cmd = [
        _tool("ffprobe"), "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as error:
        raise FFmpegError(f"ffprobe failed for {path}: {error.stderr.strip()}") from error
    data = json.loads(out)

    video_stream = None
    has_audio = False
    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video" and video_stream is None:
            video_stream = stream
        elif kind == "audio":
            has_audio = True
    if video_stream is None:
        raise FFmpegError(f"No video stream found in {path}")

    fmt = data.get("format", {})
    duration = 0.0
    for candidate in (video_stream.get("duration"), fmt.get("duration")):
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    fps = _parse_fraction(video_stream.get("avg_frame_rate") or "0") \
        or _parse_fraction(video_stream.get("r_frame_rate") or "0")

    nb_frames = 0
    try:
        nb_frames = int(video_stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        nb_frames = 0
    if nb_frames <= 0 and duration > 0 and fps > 0:
        nb_frames = round(duration * fps)

    return VideoInfo(
        path=path,
        duration=duration,
        fps=fps,
        nb_frames=nb_frames,
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        has_audio=has_audio,
    )


def list_videos(folder: str | Path, recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FFmpegError(f"Source folder does not exist or is not a directory: {folder}")
    globber = folder.rglob("*") if recursive else folder.glob("*")
    files = [p for p in globber if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(files)


def _scale_filter(width: int | None, height: int | None) -> str | None:
    if not width and not height:
        return None
    w = width if width else -2
    h = height if height else -2
    # -2 keeps the dimension divisible by 2 (required by yuv420p) while
    # preserving aspect ratio for the unspecified side.
    return f"scale={w}:{h}:flags=lanczos"


def cut_exact_clip(
    source: str | Path,
    start: float,
    n_frames: int,
    out_path: str | Path,
    *,
    fps: float = 24.0,
    width: int | None = None,
    height: int | None = None,
    crf: int = 18,
    keep_audio: bool = True,
    audio_sample_rate: int = 32000,
    audio_channels: int = 2,
    overwrite: bool = True,
) -> Path:
    """Cut exactly `n_frames` video frames at constant `fps` into `out_path`.

    `-vf fps=` forces constant frame rate (dropping/duplicating as needed) and
    `-frames:v n_frames` makes ffmpeg stop after exactly that many frames, so
    the result is guaranteed to satisfy the H3 frame count. `-ss` before `-i`
    gives fast input seeking; because the clip is re-encoded (not stream-copied),
    ffmpeg decodes from the preceding keyframe and discards up to the requested
    timestamp, so the start is frame-accurate (needed for speech-aware cuts).

    Audio (when kept) is re-encoded to AAC at `audio_sample_rate` /
    `audio_channels` in the SAME pass as the video, so the clip's A/V stay in
    sync. H3-Base expects 32 kHz stereo; pass 0 for either value to inherit the
    source instead.
    """
    source = Path(source)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vf_parts = [f"fps={fps}"]
    scale = _scale_filter(width, height)
    if scale:
        vf_parts.append(scale)
    vf = ",".join(vf_parts)

    cmd = [
        _tool("ffmpeg"),
        "-y" if overwrite else "-n",
        "-ss", f"{max(0.0, start):.3f}",
        "-i", str(source),
        "-vf", vf,
        "-frames:v", str(int(n_frames)),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
    ]
    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
        if audio_sample_rate and audio_sample_rate > 0:
            cmd += ["-ar", str(int(audio_sample_rate))]
        if audio_channels and audio_channels > 0:
            cmd += ["-ac", str(int(audio_channels))]
    else:
        cmd += ["-an"]
    cmd.append(str(out_path))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as error:
        raise FFmpegError(
            f"ffmpeg failed cutting clip from {source} @ {start:.3f}s: {error.stderr.strip()[-800:]}"
        ) from error
    if not out_path.is_file():
        raise FFmpegError(f"ffmpeg reported success but no output was written: {out_path}")
    return out_path


def normalize_source(
    source: str | Path,
    out_path: str | Path,
    *,
    fps: float = 24.0,
    width: int | None = None,
    height: int | None = None,
    crf: int = 18,
    keep_audio: bool = True,
    audio_sample_rate: int = 32000,
    audio_channels: int = 2,
    overwrite: bool = True,
) -> Path:
    """Re-encode a WHOLE source to the H3 timebase before segmentation.

    Produces one constant-`fps` (optionally rescaled) intermediate so that scene
    detection, duration math and cutting all operate on a uniform timebase. This
    is the heavyweight ``normalize_source`` path -- it re-encodes the entire
    (possibly large) source once; clips are then cut from this intermediate.
    """
    source = Path(source)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vf_parts = [f"fps={fps}"]
    scale = _scale_filter(width, height)
    if scale:
        vf_parts.append(scale)
    vf = ",".join(vf_parts)

    cmd = [
        _tool("ffmpeg"),
        "-y" if overwrite else "-n",
        "-i", str(source),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
    ]
    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
        if audio_sample_rate and audio_sample_rate > 0:
            cmd += ["-ar", str(int(audio_sample_rate))]
        if audio_channels and audio_channels > 0:
            cmd += ["-ac", str(int(audio_channels))]
    else:
        cmd += ["-an"]
    cmd.append(str(out_path))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as error:
        raise FFmpegError(
            f"ffmpeg failed normalizing {source}: {error.stderr.strip()[-800:]}"
        ) from error
    if not out_path.is_file():
        raise FFmpegError(f"ffmpeg reported success but no normalized output was written: {out_path}")
    return out_path


def extract_audio_wav(source: str | Path, out_wav: str | Path, sample_rate: int = 16000) -> Path | None:
    """Extract mono 16 kHz PCM wav for transcription. Returns None if no audio."""
    source = Path(source)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _tool("ffmpeg"), "-y",
        "-i", str(source),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "wav",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_wav.is_file():
        return None
    return out_wav

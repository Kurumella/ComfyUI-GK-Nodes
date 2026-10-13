"""GK H3 Dataset Builder - the main orchestrator node.

Folder of large videos  ->  (H3-valid clips + matching .txt T2VA prompts)
in the user-selected target directory, plus a metadata.jsonl manifest.

Pipeline per source video:
  1. probe        - duration / fps / size
  2. segment      - scene-detect (+ snap) / fixed-window / single-clip -> ClipSpecs
  3. cut          - ffmpeg to EXACTLY 17*n+5 frames @ 24 fps
  4. transcribe   - Whisper -> verbatim dialogue + timings (optional)
  5. caption      - local Qwen3.8 VLM -> T2VA fields (optional; template fallback)
  6. write        - clip_XXXX.mp4 + clip_XXXX.txt + manifest row
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

from ..core import downloader, ffmpeg_utils, segment, t2va_format
from ..core.h3_frames import (
    H3_FPS,
    MAX_N,
    MIN_N,
    describe_limits,
    duration_for_n,
    frames_for_n,
)
from ..core.transcribe import (
    WHISPER_MODELS,
    slice_transcription,
    speech_boundary_spans,
    transcribe_media,
)

LOGGER = logging.getLogger("ComfyUI.GKNodes.builder")

try:  # ComfyUI node-progress bar; absent when running outside ComfyUI.
    from comfy.utils import ProgressBar
except Exception:  # pragma: no cover - exercised only outside ComfyUI
    ProgressBar = None

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TEMPDEL = PACKAGE_ROOT.parent / "tempdel"          # scratch, outside the package
WORK_DIR = TEMPDEL / "work"
DOWNLOAD_DIR = WORK_DIR / "downloads"              # yt-dlp source cache


def _language_choices() -> list[str]:
    try:
        import whisper
        return ["auto"] + [s.title() for s in sorted(whisper.tokenizer.LANGUAGES.values())]
    except Exception:
        return ["auto", "English"]


def _model_choices() -> tuple[list[str], list[str]]:
    try:
        from ..core.caption_engine import model_options
        return model_options()
    except Exception:
        return (["auto"], ["auto"])


class GKH3DatasetBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        weights, projectors = _model_choices()
        return {
            "required": {
                "source_folder": ("STRING", {"default": "", "tooltip": "Folder containing the large source video clips."}),
                "target_folder": ("STRING", {"default": "", "tooltip": "Output dataset folder (clips + .txt prompts are written here)."}),
                "segmentation": (["scene_detect", "fixed_window", "single_clip", "speech_aware"], {"default": "scene_detect", "tooltip": "speech_aware aligns clip boundaries to Whisper speech/sentence boundaries (needs enable_transcription); auto-falls back to fixed_window when there is no speech."}),
                "min_n": ("INT", {"default": MIN_N, "min": MIN_N, "max": MAX_N, "step": 1, "tooltip": "Minimum H3 block count (n). n=7 -> 124 frames (~5.17s)."}),
                "max_n": ("INT", {"default": MAX_N, "min": MIN_N, "max": MAX_N, "step": 1, "tooltip": "Maximum H3 block count (n). n=21 -> 362 frames (~15.08s)."}),
                "scene_threshold": ("FLOAT", {"default": 27.0, "min": 1.0, "max": 100.0, "step": 0.5, "tooltip": "PySceneDetect content threshold (lower = more cuts)."}),
                "resize_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 2, "tooltip": "0 = keep source width (aspect-preserved from height)."}),
                "resize_height": ("INT", {"default": 768, "min": 0, "max": 8192, "step": 2, "tooltip": "H3-Base is 768p. 0 = keep source height."}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1, "tooltip": "x264 quality (lower = higher quality / larger files)."}),
                "keep_audio": ("BOOLEAN", {"default": True}),
                "enable_transcription": ("BOOLEAN", {"default": True}),
                "whisper_model": (WHISPER_MODELS, {"default": "large-v3"}),
                "language": (_language_choices(), {"default": "auto"}),
                "enable_captioning": ("BOOLEAN", {"default": True, "tooltip": "Use the local Qwen3.8 VLM to write T2VA prompts. If unavailable, falls back to a template."}),
                "llm_model": (weights, {"default": weights[0]}),
                "vision_model": (projectors, {"default": projectors[0]}),
                "max_sample_frames": ("INT", {"default": 12, "min": 2, "max": 48, "step": 1, "tooltip": "Frames sampled per clip for the VLM contact sheet."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max_tokens": ("INT", {"default": 4096, "min": 256, "max": 32768, "step": 128}),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "Overwrite existing clips/prompts in the target folder."}),
                "dry_run": ("BOOLEAN", {"default": False, "tooltip": "Plan clips and report counts without cutting or writing anything."}),
            },
            "optional": {
                "source_urls": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional video URLs to download with yt-dlp, one per line (blank lines and '#' comments ignored). Used in addition to source_folder."}),
                "url_max_height": ("INT", {"default": 0, "min": 0, "max": 4320, "step": 1, "tooltip": "Max download height for URL sources. 0 = best available quality."}),
                "allow_playlists": ("BOOLEAN", {"default": False, "tooltip": "If a URL is a playlist/channel, download all its entries (otherwise just the single video)."}),
                "keep_downloads": ("BOOLEAN", {"default": False, "tooltip": "Keep downloaded source files in tempdel/work/downloads (with a yt-dlp archive) for reuse. Off = delete them after the build."}),
                "style_hint": ("STRING", {"default": "", "multiline": True, "tooltip": "Style/content guidance passed to the captioner (e.g. 'cinematic, moody')."}),
                "recursive": ("BOOLEAN", {"default": False, "tooltip": "Recurse into sub-folders when scanning the source folder."}),
                "force_unload_model": ("BOOLEAN", {"default": True, "tooltip": "Release the VLM (VRAM) when the build finishes."}),
                "trigger_word": ("STRING", {"default": "", "tooltip": "Optional LoRA trigger/activation token prepended to every clip's description (right after [Shot 1]). Blank = off."}),
                # New inputs appended at the end so existing saved workflows keep
                # their widget ordering (ComfyUI maps widgets_values positionally).
                "audio_sample_rate": ("INT", {"default": 32000, "min": 0, "max": 192000, "step": 1000, "tooltip": "Audio sample rate (Hz) for clips. H3-Base is 32000. 0 = keep source."}),
                "audio_channels": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1, "tooltip": "Audio channel count for clips. 2 = stereo (H3-Base). 0 = keep source."}),
                "normalize_source": ("BOOLEAN", {"default": False, "tooltip": "Re-encode each whole source to CFR 24fps (+ target size/audio) BEFORE segmenting, for a guaranteed-uniform intermediate. Off by default; adds a full extra encode pass per source."}),
                "fast_mode": ("BOOLEAN", {"default": False, "tooltip": "Quick smoke-test: produce exactly ONE shortest-valid H3 clip (n=7, ~5.17s) from the FIRST source only. Overrides segmentation and min_n/max_n. Off by default."}),
                "auto_download_models": ("BOOLEAN", {"default": False, "tooltip": "If the Qwen3.8 captioning GGUFs are missing, download the defaults (~13.5 GB, one time) from Hugging Face into models/LLM/Qwen3.8/. Off by default. Whisper models always auto-download on first use. Does not install the llama.cpp runtime."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("summary", "dataset_dir", "manifest_path", "clip_count")
    FUNCTION = "build"
    CATEGORY = "GK-Nodes/Dataset"
    OUTPUT_NODE = True

    # --------------------------------------------------------------------- #
    def build(
        self,
        source_folder, target_folder, segmentation, min_n, max_n, scene_threshold,
        resize_width, resize_height, crf, keep_audio, enable_transcription,
        whisper_model, language, enable_captioning, llm_model, vision_model,
        max_sample_frames, seed, max_tokens, overwrite, dry_run,
        source_urls="", url_max_height=0, allow_playlists=False, keep_downloads=False,
        style_hint="", recursive=False, force_unload_model=True, trigger_word="",
        audio_sample_rate=32000, audio_channels=2, normalize_source=False,
        fast_mode=False, auto_download_models=False,
    ):
        started = time.perf_counter()
        log: list[str] = [describe_limits(H3_FPS)]

        def note(message: str) -> None:
            """Record a line in the returned summary and echo it to the console
            so progress is visible live while the node runs."""
            log.append(message)
            LOGGER.info(message)

        if not ffmpeg_utils.ffmpeg_available():
            msg = "ffmpeg/ffprobe not found on PATH. Cannot build dataset."
            return (msg, "", "", 0)

        if min_n > max_n:
            min_n, max_n = max_n, min_n

        # Gather sources: local folder (if given) + any URLs downloaded with yt-dlp.
        videos: list[Path] = []
        source_url_map: dict[str, str] = {}
        downloaded_paths: list[Path] = []

        folder_str = (source_folder or "").strip()
        if folder_str:
            try:
                videos.extend(ffmpeg_utils.list_videos(Path(folder_str).expanduser(), recursive=recursive))
            except ffmpeg_utils.FFmpegError as error:
                log.append(f"Source folder: {error}")

        urls = downloader.parse_url_list(source_urls)
        if urls:
            if not downloader.yt_dlp_available():
                log.append("yt-dlp not installed; URL sources skipped. Install with: pip install yt-dlp")
            else:
                log.append(f"Downloading {len(urls)} URL source(s) into {DOWNLOAD_DIR} ...")
                result = downloader.download_urls(
                    urls, DOWNLOAD_DIR, max_height=url_max_height,
                    allow_playlists=allow_playlists, use_archive=keep_downloads,
                )
                log.extend(result.log)
                downloaded_paths = [p for p in result.paths if p.is_file()]
                videos.extend(downloaded_paths)
                for raw_path, url in result.path_to_url.items():
                    source_url_map[str(Path(raw_path).resolve())] = url

        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique_videos: list[Path] = []
        for v in videos:
            key = str(v.resolve())
            if key not in seen:
                seen.add(key)
                unique_videos.append(v)
        videos = unique_videos

        if fast_mode and videos:
            # Quick smoke-test: one shortest-valid H3 clip from the first source.
            # single_clip + min_n=max_n=MIN_N yields exactly one n=7 clip from the
            # start. Whisper/VLM settings are left as the user set them.
            videos = videos[:1]
            segmentation = "single_clip"
            min_n = max_n = MIN_N
            log.append(
                f"FAST mode: one {frames_for_n(MIN_N)}-frame clip "
                f"(n={MIN_N}, ~{duration_for_n(MIN_N):.2f}s) from {videos[0].name} only."
            )

        if not videos:
            self._cleanup_downloads(downloaded_paths, keep_downloads)
            return ("No source videos found. Provide a source_folder with videos "
                    "and/or source_urls to download.\n" + "\n".join(log), "", "", 0)

        dataset_dir = Path(target_folder).expanduser()
        width = resize_width or None
        height = resize_height or None

        # Plan everything first (cheap) so a dry run is useful. Each plan entry
        # carries the original source (for naming/manifest), the path to cut from
        # (the normalized intermediate when normalize_source is on), the clips,
        # and the whole-source transcription (speech_aware only, else None).
        plan: list[tuple[Path, Path, list, object]] = []
        normalized_paths: list[Path] = []
        total_clips = 0
        n_sources = len(videos)
        for src_index, video in enumerate(videos, 1):
            note(f"[plan {src_index}/{n_sources}] {video.name}")
            cut_src = video
            # Normalizing is a full extra encode, so skip it during dry runs
            # (planning only needs duration/fps, which the raw probe provides).
            if normalize_source and not dry_run:
                try:
                    norm_dir = WORK_DIR / "normalized"
                    norm_dir.mkdir(parents=True, exist_ok=True)
                    norm_out = norm_dir / f"{_safe_stem(video.stem)}_h3.mp4"
                    cut_src = ffmpeg_utils.normalize_source(
                        video, norm_out, fps=H3_FPS, width=width, height=height,
                        crf=crf, keep_audio=keep_audio,
                        audio_sample_rate=audio_sample_rate, audio_channels=audio_channels,
                    )
                    normalized_paths.append(cut_src)
                    log.append(f"normalized {video.name} -> {norm_out.name} (CFR {H3_FPS:g}fps)")
                except ffmpeg_utils.FFmpegError as error:
                    log.append(f"normalize failed for {video.name} ({error}); using source as-is.")
                    cut_src = video
            try:
                info = ffmpeg_utils.probe(cut_src)
            except ffmpeg_utils.FFmpegError as error:
                log.append(f"SKIP {video.name}: {error}")
                continue

            # speech_aware needs the whole-source transcript BEFORE segmenting.
            # Transcribing the source once (not per clip) is also reused for
            # captioning via slicing, so it is cheaper than the per-clip path.
            src_trans = None
            speech_spans = None
            if segmentation == "speech_aware" and enable_transcription:
                note(f"  transcribing {video.name} ({info.duration:.1f}s) with Whisper '{whisper_model}' ...")
                src_trans = transcribe_media(
                    str(cut_src), model_name=whisper_model, language=language,
                )
                speech_spans = speech_boundary_spans(src_trans)
                if speech_spans:
                    log.append(f"{video.name}: transcript -> {len(speech_spans)} speech span(s).")

            clips = segment.plan_clips(
                info, strategy=segmentation, min_n=min_n, max_n=max_n,
                fps=H3_FPS, scene_threshold=scene_threshold,
                speech_spans=speech_spans,
            )
            plan.append((video, cut_src, clips, src_trans))
            total_clips += len(clips)
            note(f"{video.name}: {info.duration:.1f}s -> {len(clips)} clip(s)")

        if dry_run:
            self._cleanup_downloads(downloaded_paths, keep_downloads)
            summary = (f"[DRY RUN] {len(videos)} source video(s) -> {total_clips} H3 clip(s).\n"
                       + "\n".join(log))
            return (summary, str(dataset_dir), "", total_clips)

        if total_clips == 0:
            self._cleanup_downloads(downloaded_paths, keep_downloads)
            self._cleanup_downloads(normalized_paths, False)
            return ("No H3-valid clips could be produced from the source videos.\n"
                    + "\n".join(log), str(dataset_dir), "", 0)

        dataset_dir.mkdir(parents=True, exist_ok=True)
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_dir / "metadata.jsonl"

        # Spin up the captioner lazily (one resident server for all clips).
        engine = None
        caption_error = None
        if enable_captioning:
            if auto_download_models:
                try:
                    from ..core.caption_engine import download_default_models, models_present
                    if models_present():
                        note("auto_download_models: Qwen3.8 captioning GGUFs already present.")
                    else:
                        note("auto_download_models: captioning GGUFs not found; downloading "
                             "defaults (~13.5 GB, one time) into models/LLM/Qwen3.8/ ...")
                        download_default_models(progress=note)
                except Exception as error:
                    note(f"model download failed ({error}); using template prompts if models stay missing.")
            try:
                from ..core.caption_engine import CaptionEngine
                engine = CaptionEngine(llm_model=llm_model, vision_model=vision_model)
                log.append(f"Captioner: local Qwen3.8 VLM ({engine.model_name}).")
            except Exception as error:
                caption_error = str(error)
                log.append(f"Captioner unavailable ({caption_error}); using template prompts.")

        written = 0
        done = 0
        manifest_rows: list[dict] = []
        clip_pbar = ProgressBar(total_clips) if ProgressBar else None
        try:
            for video, cut_src, clips, src_trans in plan:
                for idx, spec in enumerate(clips, 1):
                    done += 1
                    if clip_pbar is not None:
                        clip_pbar.update(1)
                    stem = f"{_safe_stem(video.stem)}_{idx:03d}"
                    clip_path = dataset_dir / f"{stem}.mp4"
                    txt_path = dataset_dir / f"{stem}.txt"
                    note(
                        f"[clip {done}/{total_clips}] {clip_path.name} <- {video.name} "
                        f"(n={spec.n}, {spec.frames}f, start={spec.start:.2f}s, {spec.duration:.2f}s)"
                    )
                    if clip_path.exists() and not overwrite:
                        log.append(f"EXISTS (skip) {clip_path.name}")
                        continue
                    try:
                        ffmpeg_utils.cut_exact_clip(
                            cut_src, spec.start, spec.frames, clip_path,
                            fps=H3_FPS, width=width, height=height, crf=crf,
                            keep_audio=keep_audio,
                            audio_sample_rate=audio_sample_rate,
                            audio_channels=audio_channels,
                            overwrite=True,
                        )
                    except ffmpeg_utils.FFmpegError as error:
                        note(f"FAIL cut {clip_path.name}: {error}")
                        continue

                    transcription = None
                    if src_trans is not None:
                        # speech_aware: reuse the whole-source transcript, sliced
                        # to this clip's window (no re-transcription).
                        transcription = slice_transcription(
                            src_trans, spec.start, spec.end,
                        )
                    elif enable_transcription and keep_audio:
                        transcription = transcribe_media(
                            str(clip_path), model_name=whisper_model, language=language,
                        )

                    prompt_text = ""
                    if engine is not None:
                        try:
                            prompt_text = engine.caption_clip(
                                str(clip_path), spec.duration,
                                transcription=transcription, style_hint=style_hint,
                                max_sample_frames=max_sample_frames,
                                seed=seed, max_tokens=max_tokens,
                            )
                            prompt_text = t2va_format.normalize(prompt_text)
                        except Exception as error:
                            log.append(f"caption fallback for {stem}: {error}")
                            prompt_text = ""
                    if not prompt_text.strip():
                        prompt_text = t2va_format.build_template_prompt(
                            spec.duration, transcription=transcription, style_hint=style_hint,
                        )

                    if trigger_word.strip():
                        prompt_text = t2va_format.apply_trigger_word(prompt_text, trigger_word)

                    validation = t2va_format.validate(prompt_text)
                    txt_path.write_text(prompt_text, encoding="utf-8")

                    row = {
                        "clip": clip_path.name,
                        "prompt": txt_path.name,
                        "source": video.name,
                        "source_url": source_url_map.get(str(video.resolve()), ""),
                        "start": round(spec.start, 3),
                        "n": spec.n,
                        "frames": spec.frames,
                        "duration": round(spec.duration, 3),
                        "fps": H3_FPS,
                        "language": getattr(transcription, "language", "") if transcription else "",
                        "has_speech": bool(getattr(transcription, "has_speech", False)) if transcription else False,
                        "trigger_word": trigger_word.strip(),
                        "t2va_valid": validation.ok,
                    }
                    manifest_rows.append(row)
                    written += 1
                    note(f"  OK {clip_path.name} (n={spec.n}, {spec.frames}f)"
                         + ("" if validation.ok else " [T2VA warnings]"))
        except Exception:
            log.append("UNEXPECTED ERROR:\n" + traceback.format_exc())
        finally:
            if engine is not None and force_unload_model:
                engine.release()

        if manifest_rows:
            with manifest_path.open("w", encoding="utf-8") as handle:
                for row in manifest_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        removed = self._cleanup_downloads(downloaded_paths, keep_downloads)
        if removed:
            log.append(f"Removed {removed} downloaded source file(s) (keep_downloads=False).")
        norm_removed = self._cleanup_downloads(normalized_paths, False)
        if norm_removed:
            log.append(f"Removed {norm_removed} normalized intermediate(s).")

        elapsed = time.perf_counter() - started
        summary = (
            f"Dataset build complete in {elapsed:.1f}s.\n"
            f"Sources: {len(videos)} | Clips written: {written}/{total_clips}\n"
            + (f"Trigger word: {trigger_word.strip()}\n" if trigger_word.strip() else "")
            + f"Dataset dir: {dataset_dir}\n"
            f"Manifest: {manifest_path if manifest_rows else '(none)'}\n"
            + ("\n".join(log))
        )
        return (summary, str(dataset_dir), str(manifest_path) if manifest_rows else "", written)

    @staticmethod
    def _cleanup_downloads(paths, keep: bool) -> int:
        """Delete this-run downloaded source files unless the user kept them."""
        if keep or not paths:
            return 0
        removed = 0
        for path in paths:
            try:
                if path.is_file():
                    path.unlink()
                    removed += 1
            except OSError:
                pass
        return removed


def _safe_stem(stem: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return keep.strip("_") or "clip"

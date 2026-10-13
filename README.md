# ComfyUI-GK-Nodes

Prepare a **MiniMax H3** finetuning dataset from a folder of large video clips.
Point the node at a folder of long videos and a target directory; it emits
**H3-frame-conformant clips** each paired with a `.txt` prompt in the official
**T2VA** format, plus a `metadata.jsonl` manifest.

## Nodes

| Node | Category | Purpose |
|------|----------|---------|
| **GK H3 Dataset Builder** | `GK-Nodes/Dataset` | Folder of videos → dataset (clips + T2VA `.txt` + manifest). |
| **GK H3 Frame Calculator** | `GK-Nodes/Utility` | Convert freely between `n`, frame count, duration; tile a duration. |
| **GK T2VA Prompt Validator** | `GK-Nodes/Utility` | Validate / normalize a prompt against the T2VA structure. |

## MiniMax H3 frame formula

```
total_frames = 17 * n + 5
```

At 24 fps, `n ∈ [7, 21]`:

- **n = 7** → 124 frames (~5.17 s) — minimum
- **n = 21** → 362 frames (~15.08 s) — maximum

Every clip the builder writes is re-encoded to **constant 24 fps** with an
**exact** `17·n+5` frame count, so it is directly usable for H3 training.

## Output encoding (H3-Base defaults)

Clips are cut in a single ffmpeg pass that matches the **H3-Base** reference
spec out of the box:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| frame rate | **24 fps** (CFR) | Always forced; the H3 frame formula is defined at 24 fps. |
| `resize_height` | **768** | H3-Base is 768p. `0` keeps the source height. |
| `resize_width` | **0** | `0` = derive width from height, preserving aspect (lanczos, even dims). |
| `crf` | `18` | x264 quality (lower = higher quality / larger files). |
| `keep_audio` | `True` | Mux audio in the same pass as video (A/V stay in sync). |
| `audio_sample_rate` | **32000** | 32 kHz per H3-Base. `0` keeps the source rate. |
| `audio_channels` | **2** | Stereo per H3-Base. `0` keeps the source channel count. |

So with default settings each clip is **768p, 24 fps CFR, 32 kHz stereo** at the
exact H3 frame count — audio and video encoded together in one pass.

### `normalize_source` (optional, off by default)

Scene detection, duration math and cutting normally run directly on each source.
Enable **`normalize_source`** to first re-encode each *whole* source to CFR
24 fps (plus the target size/audio) into `tempdel/work/normalized/`, then segment
and cut from that uniform intermediate (deleted afterward). This guarantees a
single timebase across a mixed-fps batch, at the cost of a **full extra encode
pass per source** — leave it **off** unless you need that guarantee.

## Sources: folder and/or URLs

Point the builder at a local `source_folder`, a list of `source_urls` (downloaded
with [yt-dlp](https://github.com/yt-dlp/yt-dlp)), or both:

- **`source_urls`** — one URL per line (blank lines and `#` comments ignored;
  only `http(s)://` entries are accepted).
  Each is downloaded into `tempdel/work/downloads` and fed through the same
  pipeline. Requires `pip install yt-dlp` (and ffmpeg); if yt-dlp is missing,
  URL sources are skipped with a clear message and folder sources still build.
- **`url_max_height`** — cap download resolution (`0` = best available).
- **`allow_playlists`** — download every entry of a playlist/channel URL.
- **`keep_downloads`** — keep + cache downloads (with a yt-dlp download-archive
  so reruns skip them). Off (default) deletes the downloaded sources after the
  build. The manifest records the originating `source_url` per clip.

> You are responsible for complying with each site's Terms of Service and
> copyright when downloading.

## Pipeline

1. **Scan** the source folder for videos and/or download `source_urls`.
2. **Segment** each video (see [Segmentation strategies](#segmentation-strategies)).
3. **Cut** each clip with ffmpeg to the exact H3 frame count @ 24 fps, rescaled
   to 768p with 32 kHz stereo audio (see
   [Output encoding](#output-encoding-h3-base-defaults)).
4. **Transcribe** audio with Whisper (auto language per clip) → verbatim
   dialogue + timings.
5. **Caption** with a local **Qwen3.8** VLM → the three T2VA core fields,
   weaving in the verbatim transcript as `<d>[Language] …</d>` dialogue.
6. **Write** `clip_XXXX.mp4` + `clip_XXXX.txt` into the target folder, plus
   `metadata.jsonl`. If a `trigger_word` is set, it is prepended to each
   prompt's description first (see [Trigger word](#trigger-word-optional)).

### T2VA output format

Each `.txt` contains exactly the three T2VA core fields, in order:

```
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

Per the official guide:
<https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md>

**Dialogue blocks (`<d>`).** The verbatim transcript is placed inside
`<d>[Language] …</d>` on the speaking subject, with the speaker ID, delivery and
on-screen action kept *outside* the tag. One continuous spoken line is a **single**
`<d>` block ending in terminal punctuation (`.`/`?`/`!`). A line is split across
multiple blocks **only** when it crosses a shot cut (`<scenetrans>` in both parts)
or is truncated by the clip end (`<cutoff>`). The captioner is instructed to follow
these rules, and the **validator warns** when a `<d>` block ends without terminal
punctuation and no `<cutoff>`/`<scenetrans>` marker (the tell-tale of a continuous
line wrongly split mid-sentence). This surfaces as `t2va_valid` in the manifest and
in the **GK T2VA Prompt Validator** node.

## Fast mode (quick smoke-test)

Turn on **`fast_mode`** to produce exactly **one** clip — the shortest valid H3
size (`n=7`, 124 frames, ~5.17 s) — from the **first source file only**. It
overrides `segmentation` (to `single_clip`) and `min_n`/`max_n`, so you get a
single conformant clip end-to-end (cut → transcribe → caption → write) for a
quick pipeline check without processing a whole folder. Whisper and captioning
settings are left exactly as you set them. Off by default.

> Fast mode intentionally uses the shortest **valid** clip: H3-Base supports
> 4–15 s (`17n+5` grid), so `n=7` (~5.17 s) is the floor — shorter clips would
> not be valid H3 training data.

## Segmentation strategies

Set **`segmentation`** to choose how each source is divided into H3 clips:

- **`scene_detect`** (default) — [PySceneDetect](https://www.scenedetect.com/)
  finds natural cuts; each scene is then tiled with H3 clips that never cross a
  cut. A scene must be at least one minimum clip long **in real seconds**
  (≈5.17 s), derived from the source's actual fps, so high-fps footage (e.g.
  60 fps phone clips) no longer yields too-short scenes. A source with **no
  detectable cuts** (one continuous shot) is treated as a single scene and still
  tiled. Falls back to `fixed_window` if PySceneDetect is not installed.
- **`fixed_window`** — tile the whole duration with back-to-back H3 clips. Best
  when there is no dialogue (music/ambient only), where cut placement doesn't
  matter.
- **`single_clip`** — one H3 clip from the start of the video.
- **`speech_aware`** — align clip boundaries to **speech**. The source is
  transcribed once with Whisper, then clips are tiled so each one starts at a
  speech boundary and ends as close as possible to a sentence boundary — greatly
  reducing clips that start or end mid-sentence. Requires `enable_transcription`;
  automatically **falls back to `fixed_window`** when there is no detectable
  speech. The whole-source transcript is reused (sliced per clip) for captioning,
  so this is also *cheaper* than per-clip transcription.

> **Note on `speech_aware`:** H3 clip lengths are quantized (`17n+5` frames → 15
> discrete durations from ~5.17 s to ~15.08 s), so a clip can never match a
> sentence *exactly*. Mid-sentence cuts are **minimized, not eliminated** (typical
> end-of-clip error is a fraction of a second). Inter-sentence silence may be
> dropped, so clips are not necessarily back-to-back. Transcription runs on the
> whole source up front, so `speech_aware` builds (and dry runs) spend time
> transcribing before any clip is cut.

## Trigger word (optional)

Set **`trigger_word`** to give every clip in the dataset a shared LoRA
activation token — a unique word (e.g. `gkstyle`, `sks person`) you later name
at inference to invoke the fine-tuned concept. When set, it is inserted at the
very start of each clip's described content, immediately after the mandatory
`[Shot 1]` marker:

```
integrated_multimodal_description: [Shot 1] gkstyle, A cinematic wide shot ...
```

- It applies to both VLM-written and template-fallback prompts.
- Insertion is **idempotent** — a trigger already present is not duplicated.
- Leave it **blank (default)** to disable the feature entirely.
- The chosen `trigger_word` is recorded on every row of `metadata.jsonl`.

## Requirements

- **ffmpeg / ffprobe** on `PATH`.
- `pip install -r requirements.txt` (Whisper, OpenCV, Pillow, PySceneDetect).
- **Captioning** reuses the installed llama.cpp runtime and Qwen3.8 GGUF models
  from the sibling [`ComfyUI_Qwen_H3_Prompt`](https://github.com/chflame163/ComfyUI_Qwen_H3_Prompt)
  package (models in `ComfyUI/models/LLM/Qwen3.8/`, runtime installed via its
  `install_runtime.py`). If that package or its runtime is missing, captioning
  is skipped and a deterministic template prompt is written instead.

### Models & auto-download

- **Whisper** (transcription) **auto-downloads on first use** — the chosen
  `whisper_model` is fetched into `ComfyUI/models/stt/whisper/` the first time it
  is loaded. No manual step.
- **Qwen3.8 captioning GGUFs** are *not* downloaded by default. Turn on
  **`auto_download_models`** to fetch the defaults (~13.5 GB, one time) from
  Hugging Face into `ComfyUI/models/LLM/Qwen3.8/` when they're missing:
  `mmproj-F16.gguf` (from `unsloth/Qwen3.8-27B-GGUF`) and the 12 GB abliterated
  weight `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf`. Off by default because
  of the size; progress is logged and any failure falls back to the template
  prompt. This downloads **models only** — the llama.cpp **runtime** still comes
  from the sibling package's `install_runtime.py`.

## Example workflows

Drag any file from `examples/` onto the ComfyUI canvas (or use *Workflow → Open*):

| File | Shows |
|------|-------|
| `01_full_dataset_build.json` | End-to-end build: scene-detect segmentation + Whisper + Qwen3.8 captioning → clips + T2VA `.txt` + manifest, with the summary/dataset_dir/manifest/clip_count previewed. |
| `02_dry_run_planning.json` | `dry_run` mode — reports how many H3 clips each source yields without processing. |
| `03_frame_calculator.json` | The 17·n+5 math (here tiling a 40 s span into back-to-back H3 clips). |
| `04_t2va_prompt_validator.json` | Validate/normalize a T2VA prompt (field order, `[Shot 1]`, `<d>` dialogue-block rules, cut timestamps); edit the sample to see warnings. |
| `05_url_sources.json` | Download sources from URLs with yt-dlp (`source_urls`) and build the dataset. |
| `06_speech_aware_segmentation.json` | `speech_aware` segmentation — align clip cuts to dialogue/sentence boundaries via Whisper (falls back to `fixed_window` when there's no speech). |

In `01`/`02`/`05`/`06`, set `target_folder` (and `source_folder`/`source_urls`) to real values before running.

## Notes

- Progress is reported two ways while the node runs: the ComfyUI node shows a
  **progress bar** that advances per clip across the whole build (0→100% over
  `clip_count`), and the console logs each **file as it is processed** —
  `[plan i/N] <file>` during planning (including the Whisper step for
  `speech_aware`) and `[clip k/total] <clip> <- <source> (n, frames, start)`
  during cutting/captioning, plus an `OK`/`FAIL` line per clip. The same lines
  are collected into the returned `summary`.
- Temporary/scratch files live in a sibling `tempdel/` folder, kept out of the
  shipped package.
- `dry_run` plans and reports clip counts without cutting or writing anything
  (and skips the `normalize_source` re-encode).
- `fast_mode` produces one `n=7` clip from the first source only — a quick
  end-to-end smoke-test (see [Fast mode](#fast-mode-quick-smoke-test)).
- `scenedetect` is optional; without it, segmentation falls back to
  `fixed_window`. A `scene_detect` run also falls back to `fixed_window` for any
  source whose scenes are all shorter than the minimum H3 clip (~5.17 s).
- `speech_aware` needs `enable_transcription` on; with no detectable speech it
  falls back to `fixed_window` (see [Segmentation strategies](#segmentation-strategies)).
- Output defaults match **H3-Base** (768p / 24 fps CFR / 32 kHz stereo); set
  `resize_height=0`, `audio_sample_rate=0` or `audio_channels=0` to pass source
  values through untouched.
- Transcription is unaffected by `audio_sample_rate`/`audio_channels` — Whisper
  re-decodes clips to 16 kHz mono internally.

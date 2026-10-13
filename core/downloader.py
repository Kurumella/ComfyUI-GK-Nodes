"""Download source videos from URLs with yt-dlp.

Used by the dataset builder when `source_urls` are supplied. Prefers the yt-dlp
Python API and falls back to the CLI binary. One bad URL never aborts the batch
(`ignoreerrors`); failures are logged and skipped. When caching is enabled a
yt-dlp download-archive lets reruns skip already-fetched URLs.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger("ComfyUI.GKNodes.downloader")

ARCHIVE_NAME = ".download-archive.txt"


def yt_dlp_available() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception:
        return shutil.which("yt-dlp") is not None


def parse_url_list(text: str) -> list[str]:
    """One URL per line; blank lines and '#' comments ignored; de-duplicated.

    Only ``http(s)://`` entries are accepted. This rejects anything that could
    be mistaken for a command-line flag or a local path and keeps the download
    layer limited to real remote URLs (defense in depth alongside the ``--``
    guard in the CLI fallback).
    """
    urls: list[str] = []
    seen: set[str] = set()
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not (line.startswith("http://") or line.startswith("https://")):
            continue
        if line not in seen:
            seen.add(line)
            urls.append(line)
    return urls


def _format_selector(max_height: int) -> str:
    if max_height and max_height > 0:
        h = int(max_height)
        return f"bv*[height<=?{h}]+ba/b[height<=?{h}]"
    return "bv*+ba/b"


@dataclass
class DownloadResult:
    paths: list[Path] = field(default_factory=list)
    path_to_url: dict[str, str] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)


def _collect_filepaths(info) -> list[str]:
    paths: list[str] = []
    if not info:
        return paths
    if isinstance(info.get("requested_downloads"), list):
        for item in info["requested_downloads"]:
            fp = item.get("filepath") or item.get("_filename")
            if fp:
                paths.append(fp)
    elif isinstance(info.get("entries"), list):
        for entry in info["entries"]:
            paths.extend(_collect_filepaths(entry))
    else:
        fp = info.get("filepath") or info.get("_filename")
        if fp:
            paths.append(fp)
    return paths


def _download_api(urls, dest_dir, max_height, allow_playlists, use_archive):
    import yt_dlp

    result = DownloadResult()
    options = {
        "format": _format_selector(max_height),
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "noplaylist": not allow_playlists,
        "ignoreerrors": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 5,
    }
    if use_archive:
        options["download_archive"] = str(dest_dir / ARCHIVE_NAME)

    with yt_dlp.YoutubeDL(options) as ydl:
        for url in urls:
            try:
                info = ydl.extract_info(url, download=True)
            except Exception as error:  # pragma: no cover - network dependent
                result.log.append(f"FAIL {url}: {error}")
                continue
            files = [Path(p) for p in _collect_filepaths(info) if Path(p).is_file()]
            if not files:
                result.log.append(f"SKIP {url}: nothing downloaded (already in archive, or no media)")
                continue
            for fp in files:
                result.paths.append(fp)
                result.path_to_url[str(fp)] = url
            result.log.append(f"OK {url} -> {', '.join(f.name for f in files)}")
    return result


def _download_cli(urls, dest_dir, max_height, allow_playlists, use_archive):
    binary = shutil.which("yt-dlp")
    result = DownloadResult()
    cmd = [
        binary,
        "-f", _format_selector(max_height),
        "--merge-output-format", "mp4",
        "-o", str(dest_dir / "%(id)s.%(ext)s"),
        "--ignore-errors",
        "--no-progress", "--quiet",
        "--print", "after_move:filepath",
        "--no-simulate",
        "--no-playlist" if not allow_playlists else "--yes-playlist",
    ]
    if use_archive:
        cmd += ["--download-archive", str(dest_dir / ARCHIVE_NAME)]
    # "--" ends option parsing so a URL beginning with "-" cannot be read as a
    # yt-dlp flag (argument injection, e.g. a crafted "--exec=..." entry).
    cmd.append("--")
    cmd += list(urls)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        fp = Path(line.strip())
        if fp.is_file():
            result.paths.append(fp)
            result.path_to_url[str(fp)] = ""  # CLI batch: per-URL mapping unavailable
    if proc.returncode != 0 and not result.paths:
        result.log.append(f"yt-dlp CLI failed: {proc.stderr.strip()[-500:]}")
    else:
        result.log.append(f"yt-dlp CLI downloaded {len(result.paths)} file(s)")
    return result


def download_urls(
    urls: list[str],
    dest_dir: str | Path,
    *,
    max_height: int = 0,
    allow_playlists: bool = False,
    use_archive: bool = True,
) -> DownloadResult:
    """Download `urls` into `dest_dir`. Returns resolved paths + a per-file log."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not urls:
        return DownloadResult()
    if not yt_dlp_available():
        return DownloadResult(log=[
            "yt-dlp is not installed; cannot download URL sources. "
            "Install with: pip install yt-dlp"
        ])
    try:
        import yt_dlp  # noqa: F401
        return _download_api(urls, dest_dir, max_height, allow_playlists, use_archive)
    except Exception:
        return _download_cli(urls, dest_dir, max_height, allow_playlists, use_archive)

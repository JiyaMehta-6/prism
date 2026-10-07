"""Inventory and deletion of the data PRISM keeps on this machine.

Everything PRISM stores is local; this module lists it per category and
deletes exactly those categories on request. Only the paths below are ever
enumerated, so videos, champion portraits and other assets can never be
removed by accident. The analysis log is *truncated* rather than deleted -
the running logger keeps its file handle open, and truncation works with
any handle while keeping future appends valid. Its rotated backups
(``analysis.log.1`` .. ``analysis.log.3``) are not held open and are simply
removed, so clearing the log really clears the whole history.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from core import config
from core.logger import get_logger

logger = get_logger("storage")

LAST_ANALYSIS_NAME = "last_analysis.json"


@dataclass(frozen=True)
class StorageEntry:
    """One deletable category of stored data."""

    key: str
    label: str
    detail: str
    paths: Tuple[str, ...]
    count: int
    size_bytes: int
    action: str  # "delete" (remove files) or "truncate" (empty live log, drop backups)


def _files_in(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    found: List[str] = []
    try:
        for root, _dirs, names in os.walk(directory):
            for name in names:
                found.append(os.path.join(root, name))
    except OSError:
        pass  # directory vanished mid-scan: report what was found
    return sorted(found)


def _pdf_files() -> List[str]:
    if not os.path.isdir(config.OUTPUT_DIR):
        return []
    try:
        names = os.listdir(config.OUTPUT_DIR)
    except OSError:
        return []
    return sorted(
        os.path.join(config.OUTPUT_DIR, name)
        for name in names
        if name.lower().endswith(".pdf")
        and os.path.isfile(os.path.join(config.OUTPUT_DIR, name))
    )


def _log_files() -> List[str]:
    """The live log plus its rotated backups (``analysis.log.1`` ...)."""
    found: List[str] = []
    if os.path.isfile(config.LOG_PATH):
        found.append(config.LOG_PATH)
    directory = os.path.dirname(config.LOG_PATH) or "."
    prefix = os.path.basename(config.LOG_PATH) + "."
    try:
        for name in sorted(os.listdir(directory)):
            suffix = name[len(prefix):] if name.startswith(prefix) else ""
            if suffix.isdigit() and os.path.isfile(os.path.join(directory, name)):
                found.append(os.path.join(directory, name))
    except OSError:
        pass  # log dir vanished mid-scan: report the live file only
    return found


def _size_of(paths: Sequence[str]) -> int:
    total = 0
    for path in paths:
        try:
            total += os.path.getsize(path)
        except OSError:
            continue
    return total


def _entry(key: str, label: str, detail: str, paths: Sequence[str], action: str) -> StorageEntry:
    unique = tuple(dict.fromkeys(paths))
    return StorageEntry(
        key=key,
        label=label,
        detail=detail,
        paths=unique,
        count=len(unique),
        size_bytes=_size_of(unique),
        action=action,
    )


def storage_entries() -> List[StorageEntry]:
    """Snapshot of every stored category with counts and sizes."""
    last_report = os.path.join(config.OUTPUT_DIR, LAST_ANALYSIS_NAME)
    return [
        _entry(
            "profiles",
            "Player profiles",
            config.PROFILE_DIR,
            [p for p in _files_in(config.PROFILE_DIR) if p.lower().endswith(".json")],
            "delete",
        ),
        _entry(
            "last_report",
            "Last analysis report (JSON)",
            last_report,
            [last_report] if os.path.isfile(last_report) else [],
            "delete",
        ),
        _entry("charts", "Rendered charts", config.CHART_DIR, _files_in(config.CHART_DIR), "delete"),
        _entry("pdfs", "Exported PDF reports", config.OUTPUT_DIR, _pdf_files(), "delete"),
        _entry("clips", "Exported roam clips", config.CLIPS_DIR, _files_in(config.CLIPS_DIR), "delete"),
        _entry(
            "log",
            "Analysis log + backups (cleared in place)",
            config.LOG_PATH,
            _log_files(),
            "truncate",
        ),
        _entry(
            "settings",
            "Settings (reset to defaults)",
            config.SETTINGS_PATH,
            [config.SETTINGS_PATH] if os.path.isfile(config.SETTINGS_PATH) else [],
            "delete",
        ),
    ]


def clear_entries(keys: Sequence[str]) -> Dict[str, str]:
    """Remove the requested categories.

    Returns ``{key: error}`` for every category that failed; an empty dict
    means everything succeeded. Unknown keys are ignored.
    """
    errors: Dict[str, str] = {}
    entries = {entry.key: entry for entry in storage_entries()}
    for key in dict.fromkeys(keys):
        entry = entries.get(key)
        if entry is None or entry.count == 0:
            continue
        try:
            if entry.action == "truncate":
                live_log = os.path.abspath(config.LOG_PATH)
                for path in entry.paths:
                    if not os.path.isfile(path):
                        continue
                    if os.path.abspath(path) == live_log:
                        # Works alongside the logger's open append handle;
                        # subsequent writes simply start from byte 0 again.
                        with open(path, "r+b") as handle:
                            handle.truncate(0)
                    else:
                        # Rotated backup - no handle open, safe to remove.
                        os.remove(path)
            else:
                for path in entry.paths:
                    os.remove(path)
            logger.info("Stored data cleared: %s (%d item(s))", key, entry.count)
        except OSError as exc:
            errors[key] = str(exc)
            logger.warning("Could not clear %s: %s", key, exc)
    return errors


def format_size(size_bytes: int) -> str:
    """Human-readable size, e.g. ``1.4 MB``."""
    size = float(size_bytes)
    if size < 1024:
        return f"{int(size)} B"
    for unit in ("KB", "MB", "GB"):
        size /= 1024.0
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
    return f"{size:.1f} GB"  # pragma: no cover - unreachable

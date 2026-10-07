"""Champion identity from the HUD portrait - offline and free.

The analysed player's champion portrait sits at a fixed spot in the HUD
(bottom-left corner, above the ability bar).  PRISM crops it, computes a
64-bit perceptual hash (pHash, DCT signature) and compares it against
pre-computed hashes of Riot's *free* Data Dragon champion icons.

Assets are downloaded at most once and only when ``PRISM_ALLOW_DOWNLOAD=1``
is set - Data Dragon is Riot's public static CDN, no API key and no cost -
into ``data/champions/``.  After that, identification is pure local maths and
the app stays fully offline.  If the cache is missing and downloads are not
allowed, identification returns ``None``; nothing in PRISM ever pays for
anything.

A match is only reported with a confidence that combines the best Hamming
distance with the margin to the runner-up; callers gate on >= 0.65 before
claiming a champion.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.logger import get_logger

logger = get_logger("champion_identifier")

_BASE_URL = "https://ddragon.leagueoflegends.com"
# Minimum confidence before PRISM *claims* a champion identity anywhere in
# the app (engine storage, profile aggregation, report display).
CLAIM_GATE = 0.65
# Normalised (x, y, w, h) of the champion portrait in a 16:9 recording,
# measured on a 1920x1080 capture and padded to tolerate HUD scaling.
PORTRAIT_ROI: Tuple[float, float, float, float] = (0.0016, 0.7935, 0.0359, 0.0630)
# Fraction of the padded box trimmed on each side: removes the gold frame,
# level badge and HP bar bleed before hashing.
_INNER_INSET = 0.10
_MAX_DOWNLOAD_ATTEMPTS = 2


def _allow_download() -> bool:
    return os.environ.get("PRISM_ALLOW_DOWNLOAD", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def phash(image: np.ndarray, size: int = 32, hash_size: int = 8) -> int:
    """Return the 64-bit DCT perceptual hash of a grayscale/BGR image."""
    if image is None or image.size == 0:
        return 0
    gray = image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64)

    # Orthonormal DCT-II basis (no SciPy dependency).
    n = size
    k = np.arange(n, dtype=np.float64)
    basis = np.cos(np.pi * (2.0 * k[:, None] + 1.0) * k[None, :] / (2.0 * n))
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    basis *= np.sqrt(2.0 / n)
    dct = basis @ small @ basis.T

    block = dct[:hash_size, :hash_size].flatten()
    median = float(np.median(block[1:]))
    value = 0
    for index, coefficient in enumerate(block):
        if coefficient >= median:
            value |= 1 << index
    return value


def hamming(first: int, second: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(first ^ second).count("1")


@dataclass
class ChampionMatch:
    """One identification result: display name plus confidence in [0, 1]."""

    name: str
    confidence: float


class ChampionIdentifier:
    """Match the HUD champion portrait against cached Data Dragon icons."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        if cache_dir is None:
            from core.config import CHAMPION_DIR

            cache_dir = CHAMPION_DIR
        self.cache_dir = cache_dir
        self.manifest_path = os.path.join(cache_dir, "manifest.json")
        self._hashes: Dict[str, int] = {}
        self._version = ""
        self._load_manifest()

    # ---------------------------------------------------------------- assets
    def _load_manifest(self) -> bool:
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            champions = manifest.get("champions", {})
            self._hashes = {
                str(name): int(str(entry.get("phash", "0")), 16)
                for name, entry in champions.items()
                if entry.get("phash")
            }
            self._version = str(manifest.get("version", ""))
            return bool(self._hashes)
        except (OSError, ValueError, TypeError):
            self._hashes = {}
            self._version = ""
            return False

    @property
    def available(self) -> bool:
        """True when a local icon-hash cache exists (identification possible)."""
        return bool(self._hashes)

    @property
    def version(self) -> str:
        return self._version

    def ensure_assets(self) -> bool:
        """Load the cache, downloading it once if explicitly allowed.

        Returns True when a usable cache is present.  Network access happens
        only when ``PRISM_ALLOW_DOWNLOAD`` is enabled *and* the cache is
        missing; every failure is swallowed and logged - identification is a
        nice-to-have, never a hard dependency.
        """
        if self.available:
            return True
        if not _allow_download():
            logger.info(
                "Champion icons not cached (data/champions missing). Set "
                "PRISM_ALLOW_DOWNLOAD=1 once while online to prefetch the free "
                "Data Dragon icon set; champion identification stays off until then."
            )
            return False
        return self._download_assets()

    def _download_assets(self) -> bool:
        os.makedirs(self.cache_dir, exist_ok=True)
        stage = f"{self.cache_dir}.part"
        os.makedirs(stage, exist_ok=True)
        try:
            version = self._fetch_json(f"{_BASE_URL}/api/versions.json")[0]
            payload = self._fetch_json(
                f"{_BASE_URL}/cdn/{version}/data/en_US/champion.json"
            )
            entries = payload.get("data", {})
            if not entries:
                raise ValueError("empty champion catalogue")
            manifest: Dict[str, Dict[str, str]] = {}
            downloaded = 0
            for key, info in entries.items():
                name = str(info.get("name") or key)
                url = f"{_BASE_URL}/cdn/{version}/img/champion/{key}.png"
                data = self._fetch_bytes(url)
                icon = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if icon is None:
                    logger.warning("Undecodable champion icon for %s - skipped", name)
                    continue
                filename = f"{key}.png"
                with open(os.path.join(stage, filename), "wb") as handle:
                    handle.write(data)
                manifest[name] = {"file": filename, "phash": f"{phash(icon):016x}"}
                downloaded += 1
            if not manifest:
                raise ValueError("no champion icons downloaded")
            # Publish only after a complete pass so a partial download never
            # masquerades as a valid cache.
            for filename in os.listdir(stage):
                os.replace(os.path.join(stage, filename), os.path.join(self.cache_dir, filename))
            # Stage then swap: a crash mid-write must not leave a truncated
            # manifest that would silently disable identification until the
            # next full prefetch.
            tmp_manifest = f"{self.manifest_path}.tmp"
            try:
                with open(tmp_manifest, "w", encoding="utf-8") as handle:
                    json.dump(
                        {"version": version, "count": downloaded, "champions": manifest},
                        handle,
                        indent=1,
                        sort_keys=True,
                    )
                os.replace(tmp_manifest, self.manifest_path)
            finally:
                if os.path.exists(tmp_manifest):
                    try:
                        os.remove(tmp_manifest)
                    except OSError:
                        pass
            os.rmdir(stage)
            self._load_manifest()
            logger.info(
                "Prefetched %d free Data Dragon champion icons (version %s)",
                downloaded,
                version,
            )
            return self.available
        except Exception as exc:  # network/IO never abort an analysis
            logger.warning("Champion icon prefetch failed: %s", exc)
            try:
                for filename in os.listdir(stage):
                    os.remove(os.path.join(stage, filename))
                os.rmdir(stage)
            except OSError:
                pass
            return False

    @staticmethod
    def _fetch_json(url: str) -> object:
        return json.loads(ChampionIdentifier._fetch_bytes(url).decode("utf-8"))

    @staticmethod
    def _fetch_bytes(url: str) -> bytes:
        import ssl

        # uv-managed Pythons ship without a system CA bundle; certifi (free,
        # pure-data package) provides one so the TLS handshake can verify.
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            context = ssl.create_default_context()
        request = urllib.request.Request(url, headers={"User-Agent": "PRISM/1.0"})
        for attempt in range(_MAX_DOWNLOAD_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=30, context=context) as response:
                    return response.read()
            except Exception:
                if attempt == _MAX_DOWNLOAD_ATTEMPTS - 1:
                    raise
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------ identify
    def portrait_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract the padded-then-inset champion portrait from a full frame."""
        if frame is None or frame.size == 0:
            return None
        height, width = frame.shape[:2]
        x, y, w, h = PORTRAIT_ROI
        x0, y0 = int(x * width), int(y * height)
        box_w, box_h = max(8, int(w * width)), max(8, int(h * height))
        x0, y0 = max(0, x0), max(0, y0)
        box = frame[y0 : y0 + box_h, x0 : x0 + box_w]
        if box.size == 0:
            return None
        bx, by = box.shape[1], box.shape[0]
        mx, my = int(bx * _INNER_INSET), int(by * _INNER_INSET)
        inner = box[my : by - my or None, mx : bx - mx or None]
        return inner if inner.size >= 16 else None

    def identify(self, frame: np.ndarray) -> Optional[ChampionMatch]:
        """Return the best champion match for ``frame`` or None."""
        if not self._hashes:
            return None
        crop = self.portrait_crop(frame)
        if crop is None:
            return None
        signature = phash(crop)
        ranked: List[Tuple[int, str]] = sorted(
            ((hamming(signature, digest), name) for name, digest in self._hashes.items()),
            key=lambda item: item[0],
        )
        if not ranked:
            return None
        best_distance, best_name = ranked[0]
        second_distance = ranked[1][0] if len(ranked) > 1 else 64
        # Calibrated on real recordings: true HUD-vs-icon matches land at
        # Hamming 6-12, random crops at ~12-32 but with a *tie* to the runner
        # up.  Confidence therefore multiplies a distance sigmoid by how far
        # the winner is ahead - a clear gap certifies, a tie can never pass
        # the 0.65 claim gate even when the raw distance looks small.
        base = 1.0 / (1.0 + math.exp((best_distance - 17.0) / 3.5))
        gap = max(0, second_distance - best_distance)
        factor = 0.5 + 0.5 * min(1.0, gap / 4.0)
        confidence = base * factor
        return ChampionMatch(best_name, round(min(1.0, confidence), 3))

    @staticmethod
    def aggregate(results: List[ChampionMatch]) -> Optional[ChampionMatch]:
        """Combine per-frame matches (mode of names, median of their confidences)."""
        if not results:
            return None
        votes: Dict[str, List[float]] = {}
        for match in results:
            votes.setdefault(match.name, []).append(match.confidence)
        name = max(votes, key=lambda key: (len(votes[key]), sum(votes[key])))
        confidences = sorted(votes[name])
        median = confidences[len(confidences) // 2]
        if len(votes[name]) < len(results):
            # Some sampled frames disagreed: keep the winner but temper it.
            median *= 0.85
        return ChampionMatch(name, round(min(1.0, median), 3))

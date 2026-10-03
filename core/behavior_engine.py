"""Per-video behavioural processing pipeline.

For one gameplay recording PRISM:

1. probes the file (validation),
2. locates the minimap ROI,
3. streams sampled frames through :class:`~vision.minimap_tracker.MinimapTracker`,
4. optionally reads the match clock with OCR to align video time to game time,
5. interpolates short detection gaps,
6. maps every position to a behavioural region,
7. derives transitions, roam events and per-video metrics,
8. infers deaths from marker dropouts and computes vision/event micro-metrics
   (forward time without nearby vision, reaction to scoreboard kills).

Progress is reported through an ``on_progress(percent, stage)`` callback so the
GUI thread always stays informed - and never blocked.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from analytics.awareness import reaction_after_kills, unwarded_forward_fraction
from analytics.deaths import detect_deaths
from analytics.events import extract_kill_events, kills_during_roams, own_team_kills
from core.config import Settings
from core.frame_extractor import FrameExtractor
from core.logger import StageTimer, get_logger
from core.models import (
    GameEvent,
    OcrReading,
    PositionSample,
    RoamEvent,
    VideoAnalysis,
    VideoInfo,
)
from vision.champion_identifier import CLAIM_GATE, ChampionIdentifier, ChampionMatch
from vision.confidence_manager import ConfidenceManager
from vision.minimap_tracker import MinimapTracker
from vision.ocr_engine import OCREngine
from vision.region_mapper import REGIONS, RegionMapper
from vision.roi_detector import ROIDetector
from vision.ward_detector import WardDetector

logger = get_logger("behavior_engine")

ProgressFn = Callable[[int, str], None]

LANE_REGIONS = ("Top Lane", "Mid Lane", "Bot Lane")
# Ward blooms are only believed inside vision corridors: lanes are full of
# clashing minion waves and Base is a healing glow - both are false-positive
# farms, while river/jungle/pit placement is what the Vision axis measures.
CORRIDOR_REGIONS = frozenset(
    ("River", "Blue Jungle", "Red Jungle", "Dragon Area", "Herald Area")
)
# Video times (seconds) where the HUD champion portrait is sampled for
# champion identification; spread out so one dead/greyed frame cannot decide.
CHAMPION_SAMPLE_TIMES: Tuple[float, ...] = (10.0, 50.0, 100.0)


class AnalysisCancelled(Exception):
    """Raised when the user cancels an in-progress analysis."""


def _noop_progress(_percent: int, _stage: str) -> None:
    return None


class BehaviorEngine:
    """Turn a single video into a :class:`VideoAnalysis` record."""

    def __init__(self, settings: Settings, on_progress: Optional[ProgressFn] = None) -> None:
        self.settings = settings
        self.on_progress = on_progress or _noop_progress
        self.mapper = RegionMapper(getattr(settings, "region_anchors", None) or None)
        self._should_cancel: Optional[Callable[[], bool]] = None
        self._last_stride = 1

    def _check_cancel(self) -> None:
        if self._should_cancel is not None and self._should_cancel():
            raise AnalysisCancelled()

    def analyze(
        self,
        info: VideoInfo,
        base_percent: int = 0,
        span: int = 100,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> VideoAnalysis:
        """Run the full pipeline for one video."""
        last_percent = [-1]

        def progress(p: int, s: str) -> None:
            # Monotonic within this video: an early stage must never report a
            # lower percent than a previous one (e.g. "Opening" then a small
            # tracking percent for a long video).
            value = max(last_percent[0], min(100, base_percent + int(span * p / 100)))
            last_percent[0] = value
            self.on_progress(value, s)

        self._should_cancel = should_cancel
        try:
            if not info.valid:
                raise ValueError(info.error or "Invalid video")

            with StageTimer(f"analyze:{info.name}", logger):
                progress(2, f"Opening {info.name}")
                tracker = MinimapTracker(min_confidence=self.settings.marker_min_confidence)
                roi_detector = ROIDetector(self.settings.minimap_side, self.settings.minimap_roi)
                confidence = ConfidenceManager(self.settings.ocr_min_confidence)
                # OCR serves two consumers: the match clock (optional) and the
                # kill-scoreboard events (always wanted while OCR is on).
                ocr = OCREngine(
                    confidence,
                    min_confidence=self.settings.ocr_min_confidence,
                    enabled=self.settings.ocr_enabled,
                )
                extractor = FrameExtractor(self.settings.sample_fps)
                ward_detector = (
                    WardDetector(sample_fps=self.settings.sample_fps)
                    if self.settings.ward_tracking
                    else None
                )
                # Champion identity: local icon cache, one optional free
                # prefetch, no cost and no telemetry. Absent cache -> skipped.
                champion_identifier: Optional[ChampionIdentifier] = None
                probe = ChampionIdentifier()
                if probe.ensure_assets():
                    champion_identifier = probe

                raw_samples, ocr_readings, detection_attempts, raw_wards, champion_hits = (
                    self._track_positions(
                        info, extractor, tracker, roi_detector, ocr,
                        ward_detector, champion_identifier, progress,
                    )
                )
                self._last_stride = extractor.stride
                confidence.note_failures()
                observed_count = len(raw_samples)
                self._check_cancel()

                progress(78, "Interpolating detection gaps")
                samples = self._interpolate(raw_samples)

                progress(84, "Mapping behavioural regions")
                self._assign_regions(samples)

                offset = self._estimate_clock_offset(ocr_readings)
                applied_offset = 0.0
                if offset is not None and self.settings.use_game_clock:
                    applied_offset = offset
                    for sample in samples:
                        sample.game_time = sample.video_time + offset
                samples = self._restrict_to_window(samples)
                self._check_cancel()

                progress(89, "Detecting rotations")
                own_base = self._infer_own_base(samples)
                roam_events = self._detect_roams(samples, own_base)

                progress(94, "Aggregating video statistics")
                self._check_cancel()
                region_time = self._region_time(samples)
                transitions = self._transitions(samples)
                metrics = self._compute_metrics(
                    samples, roam_events, region_time, own_base, observed_count,
                    detection_attempts,
                )

                events = self._extract_events(
                    ocr_readings, roam_events, own_base, applied_offset
                )
                if any(reading.key == "kill_score" for reading in ocr_readings):
                    # Only report kill metrics when the scoreboard was actually
                    # read: "no OCR data" must never look like "0 kills".
                    enemy_base = "Red" if own_base == "Blue" else "Blue"
                    metrics["team_kills_for"] = float(own_team_kills(events, own_base))
                    metrics["team_kills_against"] = float(own_team_kills(events, enemy_base))
                    metrics["kills_during_roams"] = float(
                        kills_during_roams(events, roam_events, own_base)
                    )

                if ward_detector is not None:
                    events.extend(
                        self._build_ward_events(raw_wards, applied_offset)
                    )
                    ward_count = sum(1 for e in events if e.kind == "ward")
                    duration_minutes = max(metrics.get("analyzed_minutes", 0.0), 1e-6)
                    metrics["ward_blooms"] = float(ward_count)
                    metrics["ward_blooms_per_min"] = ward_count / duration_minutes

                    # Vision-adjusted risk: forward time with no recent ward
                    # bloom anywhere nearby ("pushing without vision").
                    forward_mask = [
                        self.mapper.forward_bias(s.x, s.y, own_base) > 0.35
                        for s in samples
                    ]
                    unwarded = unwarded_forward_fraction(
                        samples,
                        forward_mask,
                        [(t + applied_offset, x, y) for t, x, y in raw_wards],
                    )
                    if unwarded is not None:
                        metrics["unwarded_forward_fraction"] = unwarded

                kill_times = [e.time for e in events if e.kind == "kill"]

                # Death inference (pure vision): the marker vanishes away
                # from base and the first reappearance is the own fountain;
                # scoreboard kill deltas corroborate when OCR was available.
                base_xy = (
                    self.mapper.blue_base if own_base == "Blue" else self.mapper.red_base
                )
                deaths = detect_deaths(samples, base_xy, kill_times)
                for death in deaths:
                    events.append(
                        GameEvent(
                            time=death.time,
                            kind="death",
                            team=own_base,
                            detail=(
                                f"Marker lost for {death.off_map_sec:.0f}s, "
                                "respawned at Base"
                                + (" (scoreboard confirmed)" if death.confirmed else "")
                            ),
                            confidence=round(death.confidence, 2),
                        )
                    )
                if deaths:
                    minutes = max(metrics.get("analyzed_minutes", 0.0), 1e-6)
                    metrics["deaths"] = float(len(deaths))
                    metrics["death_rate_per_min"] = len(deaths) / minutes
                    metrics["death_off_map_median"] = statistics.median(
                        [d.off_map_sec for d in deaths]
                    )
                    metrics["death_confirmed_fraction"] = (
                        sum(1 for d in deaths if d.confirmed) / len(deaths)
                    )
                    logger.info(
                        "%s: %d inferred death(s), %d scoreboard-confirmed",
                        info.name, len(deaths),
                        sum(1 for d in deaths if d.confirmed),
                    )

                # Reaction to scoreboard kills: how fast the player changes
                # speed or heading after a kill appears on the HUD.
                if kill_times:
                    eligible, latencies = reaction_after_kills(
                        samples,
                        kill_times,
                        window_end=self.settings.max_analysis_minutes * 60.0,
                    )
                    if eligible:
                        metrics["reaction_events_eligible"] = float(eligible)
                        metrics["reaction_response_rate"] = len(latencies) / eligible
                        metrics["reaction_latency_median"] = (
                            statistics.median(latencies) if latencies else 0.0
                        )
                events.sort(key=lambda e: e.time)

                # PRISM only *claims* a champion when the aggregated match
                # clears the confidence gate; otherwise the field stays empty.
                champion_name: Optional[str] = None
                champion_confidence = 0.0
                aggregated = ChampionIdentifier.aggregate(champion_hits)
                if aggregated is not None and aggregated.confidence >= CLAIM_GATE:
                    champion_name = aggregated.name
                    champion_confidence = aggregated.confidence

                analysis = VideoAnalysis(
                    info=info,
                    samples=samples,
                    roam_events=roam_events,
                    region_time=region_time,
                    transitions=transitions,
                    ocr_readings=ocr_readings,
                    events=events,
                    game_time_offset=offset,
                    detection_rate=metrics.get("detection_rate", 0.0),
                    metrics=metrics,
                    champion=champion_name,
                    champion_confidence=champion_confidence,
                )
                progress(100, f"Finished {info.name}")
                logger.info(
                    "%s: %d samples (%.0f%% detected), %d roams, %d events, own base=%s",
                    info.name, len(samples), metrics.get("detection_rate", 0.0) * 100,
                    len(roam_events), len(events), own_base,
                )
                return analysis
        finally:
            self._should_cancel = None

    def _extract_events(
        self,
        readings: Sequence[OcrReading],
        roam_events: Sequence[RoamEvent],
        own_base: str,
        offset: float,
    ) -> List[GameEvent]:
        """Build the HUD event timeline inside the analysed window."""
        if not any(reading.key == "kill_score" for reading in readings):
            return []
        return extract_kill_events(
            readings,
            own_base,
            offset=offset,
            window_end=self.settings.max_analysis_minutes * 60.0,
            roam_events=roam_events,
        )

    def _build_ward_events(
        self, raw_wards: Sequence[Tuple[float, float, float]], offset: float
    ) -> List[GameEvent]:
        """Convert confirmed blooms into corridor-only ward events.

        A spatial cooldown (45 s, 12 % of the map) suppresses
        re-confirmations: the same jungle spot lighting up again moments
        later is one placement (or one champion pausing on the same camp),
        not a new ward.
        """
        limit = self.settings.max_analysis_minutes * 60.0
        events: List[GameEvent] = []
        kept: List[Tuple[float, float, float]] = []
        for video_time, x, y in raw_wards:
            event_time = video_time + offset
            if not 0.0 <= event_time <= limit:
                continue
            region = self.mapper.classify(x, y)
            if region not in CORRIDOR_REGIONS:
                continue
            if any(
                abs(event_time - kt) < 45.0 and math.hypot(x - kx, y - ky) <= 0.12
                for kt, kx, ky in kept
            ):
                continue
            kept.append((event_time, x, y))
            events.append(
                GameEvent(
                    time=event_time,
                    kind="ward",
                    team="",
                    detail=f"Vision bloom (ward-like) in {region}",
                    confidence=0.5,
                )
            )
        return events

    def _track_positions(
        self,
        info: VideoInfo,
        extractor: FrameExtractor,
        tracker: MinimapTracker,
        roi_detector: ROIDetector,
        ocr: OCREngine,
        ward_detector: Optional[WardDetector],
        champion_identifier: Optional[ChampionIdentifier],
        progress: ProgressFn,
    ) -> Tuple[
        List[PositionSample],
        List[OcrReading],
        int,
        List[Tuple[float, float, float]],
        List[ChampionMatch],
    ]:
        samples: List[PositionSample] = []
        readings: List[OcrReading] = []
        raw_wards: List[Tuple[float, float, float]] = []
        champion_hits: List[ChampionMatch] = []
        pending_champion_times: List[float] = (
            sorted(CHAMPION_SAMPLE_TIMES) if champion_identifier is not None else []
        )
        champion_tol = max(1.0, 0.6 / max(self.settings.sample_fps, 0.05))
        attempts = 0
        last_ocr_time = -1e9
        ocr_budget = self.settings.ocr_max_calls_per_video
        window_seconds = self.settings.max_analysis_minutes * 60.0
        known_duration = info.duration_sec if info.duration_sec > 0 else window_seconds
        expected = max(1.0, min(known_duration, window_seconds) * self.settings.sample_fps)
        processed = 0
        offset_estimate = 0.0
        tolerance = 2.0 / max(self.settings.sample_fps, 0.1)
        last_player: Optional[Tuple[float, float]] = None

        for frame_index, timestamp, frame in extractor.iter_frames(info.path):
            self._check_cancel()
            # Stop when the *game* window is covered, not just the video window,
            # so a negative clock offset cannot cut the analysed span short.
            if timestamp > window_seconds - offset_estimate + tolerance:
                break

            roi = roi_detector.detect(frame)
            crop = roi_detector.crop(frame, roi)
            attempts += 1
            detection = tracker.process(crop)

            if detection is not None:
                last_player = (detection.x, detection.y)
                samples.append(
                    PositionSample(
                        video_time=timestamp,
                        frame_index=frame_index,
                        x=detection.x,
                        y=detection.y,
                        confidence=detection.confidence,
                        region="Unclassified",
                    )
                )

            if ward_detector is not None:
                # Carry the last known position through tracker dropouts so the
                # player's own (nearly stationary) marker cannot confirm as a
                # ward bloom while the tracker is re-acquiring.
                for bloom in ward_detector.process(crop, last_player, timestamp):
                    raw_wards.append((timestamp, bloom.x, bloom.y))

            while pending_champion_times and timestamp > pending_champion_times[0] + champion_tol:
                pending_champion_times.pop(0)  # no frame fell near this sample time
            if pending_champion_times and abs(timestamp - pending_champion_times[0]) <= champion_tol:
                pending_champion_times.pop(0)
                match = champion_identifier.identify(frame) if champion_identifier else None
                if match is not None:
                    champion_hits.append(match)

            if (
                self.settings.ocr_enabled
                and ocr_budget > 0
                and timestamp - last_ocr_time >= self.settings.ocr_interval_sec
            ):
                last_ocr_time = timestamp
                ocr_budget -= 1
                if self.settings.use_game_clock:
                    clock = ocr.read_game_clock(frame, timestamp)
                    if clock is not None:
                        seconds, conf = clock
                        readings.append(
                            OcrReading("game_timer", str(seconds), conf, timestamp)
                        )
                        offsets: List[float] = []
                        for reading in readings:
                            try:
                                offsets.append(float(reading.value) - reading.video_time)
                            except (TypeError, ValueError):
                                continue
                        if len(offsets) >= 2 and max(offsets) - min(offsets) <= 60.0:
                            # Only a *consistent* multi-reading estimate may
                            # shorten the scan; a single outlier must never
                            # truncate the scan (clamped as a second line
                            # of defence).
                            candidate = statistics.median(offsets)
                            offset_estimate = max(-600.0, min(600.0, candidate))
                        else:
                            # Readings became inconsistent: drop the provisional
                            # estimate so a stale offset cannot stop the scan
                            # early and silently lose the tail of the window.
                            offset_estimate = 0.0
                # Kill scoreboard: one pair of digits per pass feeds the event
                # timeline (cheap: ~0.15s per side with the tight ROIs).
                score = ocr.read_kill_score(frame)
                if score is not None:
                    blue, red, score_conf = score
                    readings.append(
                        OcrReading("kill_score", f"{blue}:{red}", score_conf, timestamp)
                    )


            processed += 1
            if processed % 5 == 0 or timestamp >= window_seconds:
                percent = min(76, int(76.0 * processed / expected))
                progress(percent, f"Tracking {info.name} @{timestamp:05.1f}s")

        return samples, readings, attempts, raw_wards, champion_hits

    def _interpolate(self, raw: Sequence[PositionSample]) -> List[PositionSample]:
        """Fill short detection gaps with linear interpolation."""
        if not raw:
            logger.warning("No positions detected - analysis will be limited")
            return []

        observed = list(raw)
        if len(observed) == 1:
            return observed

        max_gap = self.settings.max_interpolate_gap
        stride = max(1, self._last_stride)
        filled: List[PositionSample] = [observed[0]]

        for i in range(1, len(observed)):
            prev, curr = observed[i - 1], observed[i]
            if curr.frame_index >= 0 and prev.frame_index >= 0:
                # Frames actually skipped by the sampler (never interpolated
                # when native fps < sample fps, where stride == 1).
                gap = (curr.frame_index - prev.frame_index) // stride - 1
            else:
                gap = int(round((curr.video_time - prev.video_time) * self.settings.sample_fps)) - 1
            if 0 < gap <= max_gap and (curr.x != prev.x or curr.y != prev.y):
                for step in range(1, gap + 1):
                    ratio = step / (gap + 1)
                    filled.append(
                        PositionSample(
                            video_time=prev.video_time + ratio * (curr.video_time - prev.video_time),
                            frame_index=-1,
                            x=prev.x + ratio * (curr.x - prev.x),
                            y=prev.y + ratio * (curr.y - prev.y),
                            confidence=min(prev.confidence, curr.confidence) * 0.75,
                            region="Unclassified",
                            detected=False,
                        )
                    )
            filled.append(curr)

        filled.sort(key=lambda s: s.video_time)
        return filled

    def _assign_regions(self, samples: Sequence[PositionSample]) -> None:
        for sample in samples:
            sample.region = self.mapper.classify(sample.x, sample.y)

    def _estimate_clock_offset(self, readings: Sequence[OcrReading]) -> Optional[float]:
        offsets: List[float] = []
        for reading in readings:
            if reading.confidence < self.settings.ocr_min_confidence:
                continue
            try:
                seconds = float(reading.value)
            except (TypeError, ValueError):
                continue
            offsets.append(seconds - reading.video_time)
        if len(offsets) < 2:
            if offsets:
                logger.warning(
                    "Only one usable clock reading; ignoring it and using video time"
                )
            return None
        median = statistics.median(offsets)
        # Discard outlier readings before judging stability so a single bad
        # OCR read cannot invalidate an otherwise consistent clock.
        inliers = [value for value in offsets if abs(value - median) <= 60.0]
        if len(inliers) < 2:
            # Only one reading agrees with the majority: unusable.
            logger.warning("Game clock readings disagree; using video time")
            return None
        # Spread must be judged over *all* readings: the inlier set is bounded
        # to +/-60s of the median by construction, so its own spread could
        # never exceed 120s and the instability check would be dead code.
        spread = max(offsets) - min(offsets)
        if spread > 300:
            logger.warning("Unstable game clock (spread %.0fs); using video time", spread)
            return None
        logger.info(
            "Game clock offset estimated: %+.1fs (n=%d, %d outliers dropped)",
            median, len(offsets), len(offsets) - len(inliers),
        )
        return median

    def _restrict_to_window(self, samples: List[PositionSample]) -> List[PositionSample]:
        """Keep only samples inside the configured analysis window."""
        limit = self.settings.max_analysis_minutes * 60.0
        kept = [s for s in samples if 0.0 <= s.effective_time <= limit]
        dropped = len(samples) - len(kept)
        if dropped:
            level = logger.warning if dropped > 0.2 * len(samples) else logger.info
            level(
                "Dropped %d of %d samples outside the 0-%.0f min window",
                dropped, len(samples), limit / 60.0,
            )
        return kept

    def _infer_own_base(self, samples: Sequence[PositionSample]) -> str:
        blue, red = 0, 0
        for sample in samples:
            if sample.region != "Base":
                continue
            d_blue = math.hypot(sample.x - self.mapper.blue_base[0], sample.y - self.mapper.blue_base[1])
            d_red = math.hypot(sample.x - self.mapper.red_base[0], sample.y - self.mapper.red_base[1])
            if d_blue <= d_red:
                blue += 1
            else:
                red += 1
        if blue == red == 0:
            # No base visits: fall back to the majority half of the map
            # (blue occupies the y > x half) instead of assuming Blue.
            blue_half = sum(1 for s in samples if s.y > s.x)
            red_half = len(samples) - blue_half
            if blue_half == red_half == 0:
                return "Blue"
            return "Blue" if blue_half >= red_half else "Red"
        return "Blue" if blue >= red else "Red"

    def _detect_roams(self, samples: Sequence[PositionSample], own_base: str) -> List[RoamEvent]:
        """Detect excursions out of the player's dominant lane."""
        if not samples:
            return []

        primary = self._primary_lane(samples)
        events: List[RoamEvent] = []
        if primary is None:
            return self._detect_territory_excursions(samples, own_base)

        min_duration = self.settings.roam_min_duration_sec
        max_absence = self.settings.roam_max_lane_absence_sec
        pending: List[PositionSample] = []
        origin = primary

        def flush(end_time: float) -> None:
            nonlocal pending
            if not pending:
                return
            duration = end_time - pending[0].effective_time
            if duration >= min_duration:
                regions = [s.region for s in pending if s.region != primary]
                if regions:
                    destination = Counter(regions).most_common(1)[0][0]
                    path = _collapse_path([primary] + regions)
                    events.append(
                        RoamEvent(
                            start_time=pending[0].effective_time,
                            end_time=end_time,
                            origin_region=origin,
                            destination_region=destination,
                            path=path,
                        )
                    )
            pending = []

        for sample in samples:
            if sample.region == primary:
                flush(sample.effective_time)
            elif sample.region in ("Base",):
                flush(sample.effective_time)
            else:
                if pending and sample.effective_time - pending[0].effective_time > max_absence:
                    # Close the excursion at the *last* sample inside the
                    # absence budget: including the current sample would
                    # overshoot the limit and swallow the next event's start.
                    flush(pending[-1].effective_time)
                pending.append(sample)
        if pending:
            flush(pending[-1].effective_time)
        return events

    def _detect_territory_excursions(
        self, samples: Sequence[PositionSample], own_base: str
    ) -> List[RoamEvent]:
        """Fallback roam logic for junglers / supports without a fixed lane."""
        events: List[RoamEvent] = []
        pending: List[PositionSample] = []
        min_duration = self.settings.roam_min_duration_sec

        def flush(end_time: float) -> None:
            nonlocal pending
            if pending and end_time - pending[0].effective_time >= min_duration:
                regions = [s.region for s in pending]
                destination = Counter(regions).most_common(1)[0][0]
                events.append(
                    RoamEvent(
                        start_time=pending[0].effective_time,
                        end_time=end_time,
                        origin_region="Own Territory",
                        destination_region=destination,
                        path=_collapse_path(["Own Territory"] + regions),
                    )
                )
            pending = []

        for sample in samples:
            if self.mapper.is_enemy_side(sample.x, sample.y, own_base):
                pending.append(sample)
            else:
                flush(sample.effective_time)
        if pending:
            flush(pending[-1].effective_time)
        return events

    def _primary_lane(self, samples: Sequence[PositionSample]) -> Optional[str]:
        counts = {lane: 0 for lane in LANE_REGIONS}
        for sample in samples:
            if sample.region in counts:
                counts[sample.region] += 1
        total = sum(counts.values())
        if total == 0:
            return None
        best = max(counts, key=counts.get)  # type: ignore[arg-type]
        if counts[best] / max(1, len(samples)) < 0.10:
            return None
        return best

    def _region_time(self, samples: Sequence[PositionSample]) -> Dict[str, float]:
        times: Dict[str, float] = {region: 0.0 for region in REGIONS}
        if not samples:
            return times
        # Tail sample carries one sampling interval, not a hardcoded second
        # (sample_fps is configurable, so 1.0s would be wrong).
        tail_delta = 1.0 / max(self.settings.sample_fps, 0.1)
        ordered = sorted(samples, key=lambda s: s.video_time)
        for i in range(len(ordered)):
            sample = ordered[i]
            if i + 1 < len(ordered):
                delta = max(0.0, ordered[i + 1].video_time - sample.video_time)
            else:
                delta = tail_delta
            region = sample.region if sample.region in times else "Base"
            times[region] += delta
        return times

    @staticmethod
    def _transitions(samples: Sequence[PositionSample]) -> Dict[str, Dict[str, int]]:
        matrix: Dict[str, Dict[str, int]] = {r: {} for r in REGIONS}
        previous: Optional[str] = None
        for sample in sorted(samples, key=lambda s: s.video_time):
            region = sample.region if sample.region in matrix else None
            if region is None:
                continue
            if previous and previous != region:
                matrix[previous][region] = matrix[previous].get(region, 0) + 1
            if previous != region:
                previous = region
        return matrix

    def _compute_metrics(
        self,
        samples: Sequence[PositionSample],
        roams: Sequence[RoamEvent],
        region_time: Dict[str, float],
        own_base: str,
        observed_count: int,
        attempts: int,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        total_time = sum(region_time.values()) or 1.0
        metrics["analyzed_minutes"] = total_time / 60.0
        metrics["detection_rate"] = (observed_count / attempts) if attempts else 0.0
        metrics["observed_samples"] = float(observed_count)
        metrics["tracking_attempts"] = float(attempts)
        metrics["own_base_blue"] = 1.0 if own_base == "Blue" else 0.0

        for region, seconds in region_time.items():
            metrics[f"region_{region}"] = seconds / total_time

        lane_total = sum(region_time[r] for r in LANE_REGIONS)
        jungle_total = region_time["Blue Jungle"] + region_time["Red Jungle"]
        metrics["lane_fraction"] = lane_total / total_time
        metrics["jungle_fraction"] = jungle_total / total_time
        metrics["river_fraction"] = region_time["River"] / total_time
        metrics["base_fraction"] = region_time["Base"] / total_time
        metrics["objective_fraction"] = (
            region_time["Dragon Area"] + region_time["Herald Area"]
        ) / total_time
        metrics["vision_proxy_fraction"] = (
            region_time["River"] + 0.5 * jungle_total
        ) / total_time

        enemy = sum(
            1 for s in samples if self.mapper.is_enemy_side(s.x, s.y, own_base)
        )
        metrics["enemy_territory_fraction"] = enemy / len(samples) if samples else 0.0
        metrics["own_territory_fraction"] = 1.0 - metrics["enemy_territory_fraction"]

        forward_biases = [self.mapper.forward_bias(s.x, s.y, own_base) for s in samples]
        metrics["forward_bias_mean"] = _mean(forward_biases)
        metrics["forward_high_fraction"] = (
            sum(1 for value in forward_biases if value > 0.35) / len(forward_biases)
            if forward_biases
            else 0.0
        )

        speeds, turns = self._movement_stats(samples)
        metrics["speed_mean"] = _mean(speeds)
        metrics["speed_p90"] = _percentile(speeds, 90)
        p90 = metrics["speed_p90"]
        metrics["burst_fraction"] = (
            sum(1 for s in speeds if s >= p90 * 0.9) / len(speeds) if speeds and p90 > 0 else 0.0
        )
        metrics["erratic_turn_rate"] = turns / max(1, len(speeds))

        duration_minutes = max(total_time / 60.0, 0.01)
        metrics["roam_count"] = float(len(roams))
        metrics["roam_rate_per_min"] = len(roams) / duration_minutes
        metrics["roam_median_duration"] = (
            statistics.median([r.duration for r in roams]) if roams else 0.0
        )
        destination_counter: Dict[str, int] = {}
        for roam in roams:
            destination_counter[roam.destination_region] = (
                destination_counter.get(roam.destination_region, 0) + 1
            )
        metrics["roam_destinations_unique"] = float(len(destination_counter))

        tower_near = sum(
            1
            for s in samples
            if self.mapper.nearest_tower(s.x, s.y)[1] <= self.mapper.a["tower_radius"]
        )
        metrics["tower_presence_fraction"] = tower_near / len(samples) if samples else 0.0

        window_stats = self._window_stats(samples, own_base)
        metrics["window_forward_std"] = window_stats["forward_std"]
        metrics["window_speed_std"] = window_stats["speed_std"]
        metrics["window_burst_growth"] = window_stats["burst_growth"]
        metrics["window_count"] = window_stats["count"]
        metrics["late_enemy_fraction"] = window_stats["late_enemy_fraction"]
        return metrics

    @staticmethod
    def _movement_stats(samples: Sequence[PositionSample]) -> Tuple[List[float], int]:
        speeds: List[float] = []
        turns = 0
        previous_heading: Optional[Tuple[float, float]] = None

        for i in range(1, len(samples)):
            prev, curr = samples[i - 1], samples[i]
            dt = curr.effective_time - prev.effective_time
            if dt <= 1e-6:
                continue
            dx, dy = curr.x - prev.x, curr.y - prev.y
            distance = math.hypot(dx, dy)
            speeds.append(distance / dt)
            if distance > 1e-4:
                heading = (dx / distance, dy / distance)
                if previous_heading is not None:
                    dot = max(
                        -1.0,
                        min(
                            1.0,
                            heading[0] * previous_heading[0] + heading[1] * previous_heading[1],
                        ),
                    )
                    if math.degrees(math.acos(dot)) > 115.0:
                        turns += 1
                previous_heading = heading
        return speeds, turns

    def _window_stats(self, samples: Sequence[PositionSample], own_base: str) -> Dict[str, float]:
        empty = {
            "forward_std": 0.0,
            "speed_std": 0.0,
            "burst_growth": 0.0,
            "count": 0.0,
            "late_enemy_fraction": 0.0,
        }
        if len(samples) < 6:
            return dict(empty)

        mapper = self.mapper
        times = [s.effective_time for s in samples]
        t_min, t_max = min(times), max(times)
        if t_max - t_min < 10:
            return dict(empty)

        bucket_count = 5
        width = (t_max - t_min) / bucket_count
        forward_means: List[float] = []
        speed_means: List[float] = []
        burst_means: List[float] = []
        late_enemy = 0
        late_total = 0

        for index in range(bucket_count):
            low = t_min + index * width
            high = low + width
            if index == bucket_count - 1:
                bucket = [s for s in samples if low <= s.effective_time <= high]
            else:
                bucket = [s for s in samples if low <= s.effective_time < high]
            if len(bucket) < 3:
                continue
            biases = [mapper.forward_bias(s.x, s.y, own_base) for s in bucket]
            speeds = [
                math.hypot(
                    bucket[i].x - bucket[i - 1].x, bucket[i].y - bucket[i - 1].y
                )
                / max(1e-6, bucket[i].effective_time - bucket[i - 1].effective_time)
                for i in range(1, len(bucket))
            ]
            forward_means.append(_mean(biases))
            speed_means.append(_mean(speeds))
            burst_means.append(_percentile(speeds, 90) if speeds else 0.0)
            if index >= 3:
                # "Late" = the final 40% of the analysed window only, so the
                # metric really measures late-window exposure, not the whole
                # window (which would duplicate enemy_territory_fraction).
                for sample in bucket:
                    if mapper.is_enemy_side(sample.x, sample.y, own_base):
                        late_enemy += 1
                    late_total += 1

        if len(forward_means) < 2:
            return {
                "forward_std": 0.0,
                "speed_std": 0.0,
                "burst_growth": 0.0,
                "count": float(len(forward_means)),
                "late_enemy_fraction": 0.0,
            }

        growth = 0.0
        if len(burst_means) >= 2 and burst_means[0] > 1e-9:
            growth = (burst_means[-1] - burst_means[0]) / burst_means[0]
            # Keep the figure reportable: a near-zero first window can make the
            # raw ratio explode into meaningless percentages.
            growth = max(-1.0, min(10.0, growth))

        return {
            "forward_std": statistics.pstdev(forward_means),
            "speed_std": statistics.pstdev(speed_means),
            "burst_growth": growth,
            "count": float(len(forward_means)),
            "late_enemy_fraction": (late_enemy / late_total) if late_total else 0.0,
        }


def _collapse_path(regions: Sequence[str]) -> List[str]:
    path: List[str] = []
    for region in regions:
        if not path or path[-1] != region:
            path.append(region)
    return path


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return float(ordered[low])
    weight = index - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)



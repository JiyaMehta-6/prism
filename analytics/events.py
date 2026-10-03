"""HUD event extraction from OCR scoreboard readings.

The top-centre scoreboard prints the running team kill totals (blue left,
red right).  PRISM reads that pair on a fixed cadence; successive readings
that *increase* prove kills happened inside the interval, so the deltas are
turned into timestamped :class:`~core.models.GameEvent` records:

    (4, 6) at t=180s  ->  (5, 6) at t=210s   =>   Blue scored 1 kill

The rules are deliberately conservative:

* a decrease on either side is impossible in a real match, so it is treated
  as an OCR glitch and only re-bases the comparison (never a negative kill),
* the event is stamped at the *observing* reading - kills are only known to
  have happened at or before that moment,
* a pair of readings covering more than ``max_interval_sec`` is dropped
  because the attribution window would be too wide to be useful,
* events outside the analysis window are discarded.

Everything here is a pure function over dataclasses so it is trivially
unit-testable without video or OCR.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from core.models import GameEvent, OcrReading, RoamEvent

KILL_KEY = "kill_score"
MAX_INTERVAL_SEC = 180.0


def parse_score_pair(value: str) -> Optional[Tuple[int, int]]:
    """Parse a ``"blue:red"`` score value recorded by the OCR pass."""
    if not value or ":" not in value:
        return None
    left, _, right = value.partition(":")
    if not left.isdigit() or not right.isdigit():
        return None
    blue, red = int(left), int(right)
    if not (0 <= blue <= 80 and 0 <= red <= 80):
        return None
    return blue, red


def extract_kill_events(
    readings: Sequence[OcrReading],
    own_base: str,
    offset: float = 0.0,
    window_end: Optional[float] = None,
    roam_events: Sequence[RoamEvent] = (),
) -> List[GameEvent]:
    """Turn scoreboard readings into team-kill events (game time).

    ``offset`` maps video time to game time (``game = video + offset``);
    ``window_end`` drops anything beyond the analysed early game; roaming
    events are only used to count how many own-team kills happened while the
    player was rotating.
    """
    score_readings = sorted(
        (r for r in readings if r.key == KILL_KEY),
        key=lambda r: r.video_time,
    )
    events: List[GameEvent] = []
    previous: Optional[Tuple[int, int]] = None
    previous_time = 0.0

    for reading in score_readings:
        pair = parse_score_pair(reading.value)
        if pair is None:
            continue
        current_time = reading.video_time + offset
        if previous is None:
            previous, previous_time = pair, current_time
            continue
        delta_blue = pair[0] - previous[0]
        delta_red = pair[1] - previous[1]
        if delta_blue < 0 or delta_red < 0 or (
            current_time - previous_time > MAX_INTERVAL_SEC
        ):
            # Score went backwards (glitch / reconnect) or the gap is too
            # wide: re-base instead of inventing kills.
            previous, previous_time = pair, current_time
            continue

        for delta, team in ((delta_blue, "Blue"), (delta_red, "Red")):
            if delta <= 0:
                continue
            noun = "kill" if delta == 1 else "kills"
            events.append(
                GameEvent(
                    time=current_time,
                    kind="kill",
                    team=team,
                    detail=f"{team} team scored {delta} {noun}",
                    confidence=max(0.0, min(1.0, reading.confidence)),
                    count=delta,
                )
            )
        previous, previous_time = pair, current_time

    if window_end is not None:
        events = [e for e in events if 0.0 <= e.time <= window_end]
    return events


def own_team_kills(events: Sequence[GameEvent], own_base: str) -> int:
    """Total own-team kills across ``events``."""
    return sum(e.count for e in events if e.kind == "kill" and e.team == own_base)


def kills_during_roams(
    events: Sequence[GameEvent],
    roam_events: Sequence[RoamEvent],
    own_base: str,
) -> int:
    """Own-team kills that happened inside a roam window."""
    total = 0
    for event in events:
        if event.kind != "kill" or event.team != own_base:
            continue
        for roam in roam_events:
            if roam.start_time <= event.time <= roam.end_time:
                total += event.count
                break
    return total

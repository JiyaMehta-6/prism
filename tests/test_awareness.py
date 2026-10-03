from __future__ import annotations

from analytics.awareness import reaction_after_kills, unwarded_forward_fraction
from analytics.deaths import detect_deaths
from analytics.insights import _death_pattern, _slow_reaction, _unwarded_aggression
from core.models import Fingerprint, PlayerProfile, PositionSample

BASE = (0.85, 0.15)


def _sample(t: float, x: float, y: float, detected: bool = True) -> PositionSample:
    return PositionSample(
        video_time=t,
        frame_index=-1,
        x=x,
        y=y,
        confidence=0.9,
        region="Mid Lane",
        detected=detected,
    )


# --------------------------------------------------------------------- deaths


def test_death_detected_when_marker_returns_to_fountain() -> None:
    samples = [
        _sample(100.0, 0.40, 0.40),
        _sample(108.0, 0.85, 0.16),  # 8 s gap, reappears at base
    ]
    deaths = detect_deaths(samples, BASE, kill_times=[104.0])
    assert len(deaths) == 1
    death = deaths[0]
    assert death.time == 100.0
    assert death.off_map_sec == 8.0
    assert death.confirmed
    assert death.confidence >= 0.75  # base 0.55 + 0.25 confirmed


def test_short_unconfirmed_gap_is_not_a_death() -> None:
    """A 7 s blind gap ending at base looks like a recall - rejected."""
    samples = [_sample(50.0, 0.40, 0.40), _sample(57.0, 0.85, 0.16)]
    assert detect_deaths(samples, BASE, kill_times=[]) == []


def test_short_confirmed_gap_is_a_death() -> None:
    """With the scoreboard showing a kill, even a 7 s gap counts."""
    samples = [_sample(50.0, 0.40, 0.40), _sample(57.0, 0.85, 0.16)]
    deaths = detect_deaths(samples, BASE, kill_times=[53.0])
    assert len(deaths) == 1
    assert deaths[0].confirmed


def test_reappearance_away_from_base_is_not_a_death() -> None:
    """The tracker just blipped: they never showed up at the fountain."""
    samples = [_sample(50.0, 0.40, 0.40), _sample(62.0, 0.55, 0.55)]
    assert detect_deaths(samples, BASE, kill_times=[53.0]) == []


def test_vanishing_at_base_is_not_a_death() -> None:
    """A dropout while already at the fountain is a tracker artifact."""
    samples = [_sample(50.0, 0.86, 0.15), _sample(62.0, 0.85, 0.16)]
    assert detect_deaths(samples, BASE, kill_times=[53.0]) == []


def test_gap_beyond_max_is_ignored() -> None:
    """A 70 s absence is a recording gap, not a respawn timer."""
    samples = [_sample(50.0, 0.40, 0.40), _sample(120.0, 0.85, 0.16)]
    assert detect_deaths(samples, BASE, kill_times=[55.0]) == []


def test_only_detected_samples_anchor_gaps() -> None:
    """Interpolated fill samples never shorten a real absence."""
    samples = [
        _sample(100.0, 0.40, 0.40),
        _sample(104.0, 0.50, 0.30, detected=False),  # interpolated
        _sample(108.0, 0.85, 0.16),
    ]
    deaths = detect_deaths(samples, BASE, kill_times=[104.0])
    assert len(deaths) == 1
    assert deaths[0].off_map_sec == 8.0


# ----------------------------------------------------------- unwarded forward


def test_recent_nearby_ward_covers_forward_sample() -> None:
    samples = [_sample(100.0, 0.55, 0.45)]
    frac = unwarded_forward_fraction(samples, [True], wards=[(95.0, 0.56, 0.46)])
    assert frac == 0.0


def test_no_ward_means_fully_unwarded() -> None:
    samples = [_sample(100.0, 0.55, 0.45)]
    assert unwarded_forward_fraction(samples, [True], wards=[]) == 1.0


def test_stale_ward_does_not_cover() -> None:
    samples = [_sample(400.0, 0.55, 0.45)]
    frac = unwarded_forward_fraction(samples, [True], wards=[(200.0, 0.55, 0.45)])
    assert frac == 1.0  # 200 s old > 120 s lookback


def test_ward_outside_radius_does_not_cover() -> None:
    samples = [_sample(100.0, 0.55, 0.45)]
    frac = unwarded_forward_fraction(samples, [True], wards=[(95.0, 0.90, 0.45)])
    assert frac == 1.0


def test_mixed_coverage_yields_fraction() -> None:
    samples = [_sample(100.0, 0.55, 0.45), _sample(110.0, 0.85, 0.75)]
    frac = unwarded_forward_fraction(
        samples, [True, True], wards=[(95.0, 0.55, 0.45)]
    )
    assert frac == 0.5  # first covered, second outside the bloom radius


def test_coverage_uses_both_time_and_distance() -> None:
    sample = [_sample(400.0, 0.55, 0.45)]
    # Recent (50 s <= 120 s) and nearby -> covered.
    assert unwarded_forward_fraction(sample, [True], wards=[(350.0, 0.55, 0.45)]) == 0.0
    # Recent but far away -> uncovered.
    assert unwarded_forward_fraction(sample, [True], wards=[(350.0, 0.10, 0.90)]) == 1.0
    # Nearby but stale (250 s > 120 s) -> uncovered.
    assert unwarded_forward_fraction(sample, [True], wards=[(150.0, 0.55, 0.45)]) == 1.0


def test_no_forward_samples_returns_none() -> None:
    samples = [_sample(100.0, 0.55, 0.45)]
    assert unwarded_forward_fraction(samples, [False], wards=[]) is None


# ------------------------------------------------------------------ reaction


def test_direction_change_after_kill_is_a_response() -> None:
    samples = []
    for t in range(0, 11):  # moving east before the kill
        samples.append(_sample(float(t), 0.02 * t, 0.50))
    for t in range(11, 21):  # turns north afterwards
        samples.append(_sample(float(t), 0.20, 0.50 + 0.03 * (t - 10)))
    eligible, latencies = reaction_after_kills(samples, [10.5], window_end=60.0)
    assert eligible == 1
    assert latencies, "a 90-degree turn must register as a response"
    assert 1.0 <= latencies[0] <= 5.0


def test_no_change_means_no_response() -> None:
    samples = [_sample(float(t), 0.02 * t, 0.50) for t in range(0, 21)]
    eligible, latencies = reaction_after_kills(samples, [10.5], window_end=60.0)
    assert eligible == 1
    assert latencies == []


def test_insufficient_neighbouring_samples_is_not_eligible() -> None:
    samples = [_sample(float(t), 0.02 * t, 0.50) for t in (0, 1, 20, 21)]
    eligible, latencies = reaction_after_kills(samples, [10.5], window_end=60.0)
    assert eligible == 0
    assert latencies == []


def test_kill_outside_window_is_skipped() -> None:
    samples = [_sample(float(t), 0.02 * t, 0.50) for t in range(0, 21)]
    eligible, _ = reaction_after_kills(samples, [95.0], window_end=60.0)
    assert eligible == 0


# ------------------------------------------------------------------- insights


def _profile(**metrics: float) -> PlayerProfile:
    return PlayerProfile(
        player_label="Tester",
        video_count=3,
        total_samples=900,
        detection_rate=0.9,
        average_metrics=metrics,
    )


def test_unwarded_insight_requires_high_fraction() -> None:
    profile = _profile(unwarded_forward_fraction=0.20, forward_high_fraction=0.30)
    assert _unwarded_aggression(profile, Fingerprint(), 0.9) is None


def test_unwarded_insight_fires_with_vision_data() -> None:
    profile = _profile(unwarded_forward_fraction=0.70, forward_high_fraction=0.30)
    insight = _unwarded_aggression(profile, Fingerprint(), 0.9)
    assert insight is not None
    assert "vision" in insight.title.lower()
    assert insight.confidence > 0.3


def test_unwarded_insight_absent_without_ward_data() -> None:
    profile = _profile(forward_high_fraction=0.40)
    assert _unwarded_aggression(profile, Fingerprint(), 0.9) is None


def test_death_insight_gates() -> None:
    low = _profile(deaths=0.5, death_rate_per_min=0.05)
    assert _death_pattern(low, Fingerprint(), 0.9) is None

    high = _profile(
        deaths=3.0,
        death_rate_per_min=0.30,
        death_off_map_median=14.0,
        death_confirmed_fraction=1.0,
        unwarded_forward_fraction=0.65,
    )
    insight = _death_pattern(high, Fingerprint(), 0.9)
    assert insight is not None
    assert "death" in insight.title.lower()
    assert any("without nearby vision" in e or "vision" in e.lower() for e in insight.evidence)


def test_slow_reaction_insight_gates() -> None:
    too_few = _profile(reaction_events_eligible=2.0, reaction_response_rate=0.0)
    assert _slow_reaction(too_few, Fingerprint(), 0.9) is None

    fast = _profile(reaction_events_eligible=6.0, reaction_response_rate=0.90)
    assert _slow_reaction(fast, Fingerprint(), 0.9) is None

    slow = _profile(
        reaction_events_eligible=6.0,
        reaction_response_rate=0.33,
        reaction_latency_median=5.0,
    )
    insight = _slow_reaction(slow, Fingerprint(), 0.9)
    assert insight is not None
    assert "response" in insight.title.lower()


def test_risk_axis_optional_unwarded_indicator() -> None:
    """Risk must score identically when the unwarded metric is absent."""
    from core.fingerprint_engine import FingerprintEngine

    base = dict(
        enemy_territory_fraction=0.30,
        late_enemy_fraction=0.35,
        burst_fraction=0.25,
        erratic_turn_rate=0.20,
    )
    plain = _profile(**base)
    without = FingerprintEngine().compute(plain)
    with_metric = _profile(**base, unwarded_forward_fraction=0.80)
    boosted = FingerprintEngine().compute(with_metric)

    # Absent metric -> indicator dropped, only the classic Risk inputs speak.
    assert "unwarded" not in without.contributions["Risk"]
    assert without.get("Risk") > 0.0
    # Present metric -> the vision-adjusted share joins the axis and can
    # never lower the score.
    assert "unwarded" in boosted.contributions["Risk"]
    assert boosted.get("Risk") >= without.get("Risk")

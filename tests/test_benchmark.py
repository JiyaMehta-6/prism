from __future__ import annotations

import os


def _write_snapshot(directory, name: str, label: str, aggression: float, **extra) -> None:
    from core.models import FINGERPRINT_METRICS, save_json

    scores = {metric: 5.0 for metric in FINGERPRINT_METRICS}
    scores["Aggression"] = aggression
    scores.update(extra.get("scores", {}))
    payload = {
        "player_label": label,
        "saved_at": "20260101-000000",
        "video_count": 1,
        "fingerprint": scores,
        "region_distribution": {},
        "metrics": {},
        "vector": [0.1, 0.2, 0.3],
    }
    os.makedirs(directory, exist_ok=True)
    save_json(os.path.join(directory, f"{name}.json"), payload)


def _profile(label: str = "Current"):
    from core.models import PlayerProfile

    return PlayerProfile(
        player_label=label, video_count=1, total_samples=100, detection_rate=0.9
    )


def _fingerprint(aggression: float, **extra):
    from core.models import FINGERPRINT_METRICS, Fingerprint

    scores = {metric: 5.0 for metric in FINGERPRINT_METRICS}
    scores["Aggression"] = aggression
    scores.update(extra)
    return Fingerprint(scores=scores)


def test_benchmark_requires_three_other_profiles(tmp_path) -> None:
    from analytics.benchmark import compute_benchmark

    directory = str(tmp_path)
    for index in range(2):
        _write_snapshot(directory, f"p{index}", f"Player{index}", float(index * 2))

    result = compute_benchmark(_profile(), _fingerprint(6.0), directory)
    assert result is None


def test_benchmark_computes_midrank_percentiles(tmp_path) -> None:
    from analytics.benchmark import compute_benchmark
    from core.models import FINGERPRINT_METRICS

    directory = str(tmp_path)
    for index, value in enumerate((2.0, 4.0, 6.0, 8.0)):
        _write_snapshot(directory, f"p{index}", f"Player{index}", value)

    result = compute_benchmark(_profile(), _fingerprint(6.0), directory)
    assert result is not None
    assert result.n == 4
    # Mid-rank: two stored below, one equal => (2 + 0.5) / 4 = 62.5.
    assert abs(result.percentiles["Aggression"] - 62.5) < 1e-9
    assert len(result.notes) == len(FINGERPRINT_METRICS)
    assert "4 stored profiles" in result.summary
    assert 0.0 <= result.percentiles["Vision"] <= 100.0


def test_benchmark_excludes_own_label(tmp_path) -> None:
    from analytics.benchmark import compute_benchmark

    directory = str(tmp_path)
    for index in range(4):
        _write_snapshot(directory, f"self{index}", "Current", float(index))

    assert compute_benchmark(_profile("Current"), _fingerprint(6.0), directory) is None
    # Other players only: the gate opens again.
    for index in range(4):
        _write_snapshot(directory, f"other{index}", "Rival", float(index))
    result = compute_benchmark(_profile("Current"), _fingerprint(6.0), directory)
    assert result is not None
    assert result.n == 4


def test_benchmark_skips_incomplete_snapshots(tmp_path) -> None:
    from analytics.benchmark import compute_benchmark, load_fingerprint_snapshots
    from core.models import save_json

    directory = str(tmp_path)
    for index in range(3):
        _write_snapshot(directory, f"p{index}", f"Player{index}", float(index))
    save_json(
        os.path.join(directory, "broken.json"),
        {"player_label": "Broken", "fingerprint": {"Aggression": 5.0}},
    )
    save_json(os.path.join(directory, "notjson.json"), ["not", "a", "dict"])

    loaded = load_fingerprint_snapshots(directory)
    assert len(loaded) == 3
    assert compute_benchmark(_profile(), _fingerprint(6.0), directory) is not None

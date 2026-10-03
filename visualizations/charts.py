"""Matplotlib visualisation suite.

All charts are rendered headlessly (``Agg``), saved as PNG under
``outputs/charts`` and reused by both the GUI report tabs and the PDF
exporter.  A single dark "Hextech esports" theme (navy canvas, electric
blue, hextech gold) keeps every figure matching the GUI.
"""

from __future__ import annotations

import os
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.config import CHART_DIR
from core.logger import get_logger
from core.models import (
    FINGERPRINT_METRICS,
    Fingerprint,
    PlayerProfile,
    SimilarityResult,
    TimelineWindow,
)
from vision.region_mapper import REGIONS

logger = get_logger("charts")

# Hextech esports palette - shared with the GUI stylesheet.
plt.rcParams.update(
    {
        "figure.facecolor": "#121A2B",
        "axes.facecolor": "#121A2B",
        "savefig.facecolor": "#121A2B",
        "text.color": "#E6ECF7",
        "axes.labelcolor": "#E6ECF7",
        "axes.titlecolor": "#E6ECF7",
        "xtick.color": "#8B98AD",
        "ytick.color": "#8B98AD",
        "axes.edgecolor": "#26324A",
        "grid.color": "#2A3752",
        "grid.alpha": 0.4,
    }
)

COLORS = {
    "primary": "#3B82F6",
    "accent": "#C8AA6E",
    "green": "#37D399",
    "red": "#FF4655",
    "purple": "#A78BFA",
    "grey": "#8B98AD",
    "bg": "#121A2B",
}

REGION_COLORS: Dict[str, str] = {
    "Base": "#64748B",
    "Top Lane": "#38BDF8",
    "Mid Lane": "#FB7185",
    "Bot Lane": "#4ADE80",
    "River": "#60A5FA",
    "Blue Jungle": "#2DD4BF",
    "Red Jungle": "#FDA4AF",
    "Dragon Area": "#FBBF24",
    "Herald Area": "#C084FC",
}


def _chart_dir() -> str:
    return os.environ.get("PRISM_CHART_DIR") or CHART_DIR


def _save(fig: plt.Figure, name: str) -> str:
    target = _chart_dir()
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, name)
    # Write to a unique temp file and swap: a reader (PDF export) never sees
    # a half-written PNG, and two instances cannot collide on the temp name.
    tmp_path = f"{path}.{os.getpid()}.tmp"
    # matplotlib infers the format from the extension - the temp suffix is
    # opaque, so the real target extension must be passed explicitly.
    fmt = path.rsplit(".", 1)[-1].lower() if "." in path else "png"
    try:
        fig.savefig(tmp_path, format=fmt, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
        os.replace(tmp_path, path)
        logger.debug("Chart saved: %s", path)
        return path
    finally:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        plt.close(fig)


def _style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#26324A")


def generate_all(
    profile: PlayerProfile,
    fingerprint: Fingerprint,
    timeline: Sequence[TimelineWindow],
    similarity: Sequence[SimilarityResult],
) -> Dict[str, str]:
    """Render every chart and return a ``name -> path`` mapping."""
    paths: Dict[str, str] = {}
    generators = (
        ("fingerprint", lambda: radar_chart(fingerprint)),
        ("regions", lambda: region_chart(profile)),
        ("timeline", lambda: timeline_chart(timeline)),
        ("transitions", lambda: transition_chart(profile)),
        ("consistency", lambda: consistency_chart(profile)),
        ("similarity", lambda: similarity_chart(similarity)),
        ("insights", lambda: insights_chart(fingerprint)),
    )
    for name, fn in generators:
        try:
            path = fn()
            if path:
                paths[name] = path
        except Exception as exc:
            logger.error("Chart '%s' failed: %s", name, exc)
        finally:
            # Never leak figures when a chart errors out or returns early.
            plt.close("all")
    return paths


def radar_chart(fingerprint: Fingerprint) -> str:
    """Fingerprint radar chart (seven axes, 0-10)."""
    labels = list(FINGERPRINT_METRICS)
    values = [fingerprint.scores.get(label, 0.0) for label in labels]
    count = len(labels)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=(6.4, 6.4))
    fig.patch.set_facecolor(COLORS["bg"])
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles_closed, values_closed, color=COLORS["primary"], linewidth=2)
    ax.fill(angles_closed, values_closed, color=COLORS["primary"], alpha=0.25)
    ax.scatter(angles, values, color=COLORS["accent"], zorder=5, s=42)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="#8B98AD")
    ax.set_title("Behavioural Fingerprint", fontsize=13, fontweight="bold", pad=22)
    ax.grid(color="#2A3752")
    return _save(fig, "fingerprint_radar.png")


def region_chart(profile: PlayerProfile) -> str:
    """Region occupancy bar chart."""
    regions = [r for r in REGIONS]
    values = [profile.region_distribution.get(r, 0.0) * 100 for r in regions]
    colors = [REGION_COLORS.get(r, COLORS["grey"]) for r in regions]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.patch.set_facecolor(COLORS["bg"])
    bars = ax.barh(regions, values, color=colors, edgecolor="none")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
        )
    _style_axes(ax, "Time Spent per Region")
    ax.set_xlabel("Share of analysed time (%)")
    ax.set_xlim(0, max(values + [10]) * 1.25)
    ax.invert_yaxis()
    return _save(fig, "region_distribution.png")


def timeline_chart(timeline: Sequence[TimelineWindow]) -> str:
    """Tactical timeline: labelled windows plus aggression index."""
    if not timeline:
        return ""
    labels = [f"{w.start_label}-{w.end_label}\n{w.label}" for w in timeline]
    aggression = [w.aggression_index for w in timeline]
    confidences = [w.confidence for w in timeline]
    x = np.arange(len(timeline))

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    fig.patch.set_facecolor(COLORS["bg"])
    bars = ax.bar(x, aggression, color=COLORS["primary"], alpha=0.85, width=0.6)
    for bar, value, conf in zip(bars, aggression, confidences):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{value:.1f}\nconf {conf * 100:.0f}%",
            ha="center",
            fontsize=8,
        )
    ax.plot(x, aggression, color=COLORS["accent"], marker="o", linewidth=2, zorder=5)
    _style_axes(ax, "Tactical Timeline - Full Match Behaviour")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Aggression index (0-10)")
    ax.set_ylim(0, max(aggression + [1]) * 1.35)
    return _save(fig, "tactical_timeline.png")


def transition_chart(profile: PlayerProfile) -> str:
    """Region-to-region transition matrix heatmap."""
    # Include regions that only ever appear as a transition *target*.
    sources = [
        r
        for r in REGIONS
        if any(profile.transition_matrix.get(r, {}).values())
        or any(
            profile.transition_matrix.get(row, {}).get(r, 0)
            for row in profile.transition_matrix
        )
    ]
    if not sources:
        return ""
    matrix = np.zeros((len(sources), len(sources)))
    for i, source in enumerate(sources):
        for j, target in enumerate(sources):
            matrix[i, j] = profile.transition_matrix.get(source, {}).get(target, 0)

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    fig.patch.set_facecolor(COLORS["bg"])
    im = ax.imshow(matrix, cmap="YlGnBu")
    ax.set_xticks(range(len(sources)))
    ax.set_yticks(range(len(sources)))
    ax.set_xticklabels(sources, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(sources, fontsize=8.5)
    for i in range(len(sources)):
        for j in range(len(sources)):
            value = matrix[i, j]
            if value > 0:
                ax.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=8,
                        color="white" if value > matrix.max() * 0.6 else "#1B2631")
    ax.set_title("Region Transition Matrix", fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.75, label="transition count")
    return _save(fig, "transition_matrix.png")


def consistency_chart(profile: PlayerProfile) -> str:
    """Per-video comparison of the core behavioural indicators."""
    if len(profile.per_video_metrics) < 2:
        return ""
    names = list(profile.per_video_metrics)
    metrics = (
        ("forward_high_fraction", "Forward positioning"),
        ("lane_fraction", "Lane presence"),
        ("river_fraction", "River presence"),
        ("enemy_territory_fraction", "Enemy-side exposure"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4))
    fig.patch.set_facecolor(COLORS["bg"])
    x = np.arange(len(names))
    palette = [COLORS["primary"], COLORS["green"], COLORS["accent"], COLORS["red"]]

    for ax, ((key, title), color) in zip(axes.flat, zip(metrics, palette)):
        values = [profile.per_video_metrics[name].get(key, 0.0) * 100 for name in names]
        ax.bar(x, values, color=color, alpha=0.85)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n[:14] for n in names], rotation=30, ha="right", fontsize=7.5)
        ax.set_ylim(0, max(values + [10]) * 1.3)
        ax.grid(axis="y", alpha=0.3)
        for i, value in enumerate(values):
            ax.text(i, value + 1, f"{value:.0f}", ha="center", fontsize=7.5)

    fig.suptitle("Match-to-Match Behavioural Variation", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "consistency_variation.png")


def similarity_chart(results: Sequence[SimilarityResult]) -> str:
    """Horizontal bars of behavioural similarity against stored profiles."""
    if not results:
        return ""
    labels = [r.profile_name[:34] for r in results]
    values = [r.similarity * 100 for r in results]

    fig, ax = plt.subplots(figsize=(8.4, max(3.2, 0.6 * len(results) + 1.4)))
    fig.patch.set_facecolor(COLORS["bg"])
    colors = [COLORS["green"] if v >= 75 else COLORS["primary"] if v >= 60 else COLORS["grey"]
              for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor="none")
    for bar, value in zip(bars, values):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%",
                va="center", fontsize=9)
    _style_axes(ax, "Behavioural Profile Similarity")
    ax.set_xlabel("Cosine similarity (%)")
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    return _save(fig, "profile_similarity.png")


def insights_chart(fingerprint: Fingerprint) -> str:
    """Fingerprint comparison with confidence whiskers."""
    labels = list(FINGERPRINT_METRICS)
    values = [fingerprint.scores.get(label, 0.0) for label in labels]
    confs = [fingerprint.confidence.get(label, 0.0) * 100 for label in labels]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor(COLORS["bg"])
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=COLORS["purple"], alpha=0.9, width=0.6)
    for bar, value, conf in zip(bars, values, confs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.2,
            f"{value:.1f}\n{conf:.0f}%",
            ha="center",
            fontsize=8,
        )
    _style_axes(ax, "Fingerprint Scores with Confidence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score (0-10)")
    ax.set_ylim(0, max(values + [1]) * 1.35)
    return _save(fig, "fingerprint_bars.png")

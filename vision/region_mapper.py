"""Geometric mapping of normalised minimap coordinates to game regions.

The LoL minimap is rendered as a square with the blue base in the bottom-left
corner and the red base in the top-right corner (image coordinates, y grows
downwards).  Region classification is a purely geometric, fully explainable
pipeline evaluated in a fixed priority order:

1. Base          - proximity to a team fountain anchor
2. Objectives    - Dragon / Herald pit anchors (checked before River because
                   both pits sit inside the river band)
3. River         - central band of the anti-diagonal ``x = y``
4. Mid Lane      - band around the blue->red diagonal ``x + y = 1``
5. Top Lane      - polyline ``(i,1-i) -> (i,i) -> (1-i,i)``
6. Bot Lane      - polyline ``(i,1-i) -> (1-i,1-i) -> (1-i,i)``
7. Jungle        - residual space, split by which half of the river it falls in

where ``i = lane_inset`` (default 0.07) keeps the lanes inside the playable
area of the minimap image rather than hugging its frame.

All anchors are stored in :data:`DEFAULT_ANCHORS`.  The common scalar anchors
(lane inset, base/river/mid band widths) are editable from the settings
dialog and every scalar can be overridden through ``Settings.region_anchors``
(validated and clamped in :mod:`core.config`), so the mapper adapts to HUD
scaling or map re-skins while keeping the nine region labels stable.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]

REGIONS: List[str] = [
    "Base",
    "Top Lane",
    "Mid Lane",
    "Bot Lane",
    "River",
    "Blue Jungle",
    "Red Jungle",
    "Dragon Area",
    "Herald Area",
]

DEFAULT_ANCHORS: Dict[str, object] = {
    "blue_base": (0.05, 0.95),
    "red_base": (0.95, 0.05),
    "base_radius": 0.10,
    "dragon": (0.66, 0.70),
    "dragon_radius": 0.075,
    "herald": (0.34, 0.28),
    "herald_radius": 0.075,
    "river_start": 0.20,
    "river_end": 0.80,
    "river_threshold": 0.05,
    "mid_threshold": 0.065,
    "lane_threshold": 0.075,
    "lane_inset": 0.07,
    "towers": {
        "blue_top_outer": (0.09, 0.56),
        "blue_mid_outer": (0.30, 0.70),
        "blue_bot_outer": (0.44, 0.91),
        "red_top_outer": (0.56, 0.09),
        "red_mid_outer": (0.70, 0.30),
        "red_bot_outer": (0.91, 0.44),
    },
    "tower_radius": 0.055,
}


def _dist_point(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _dist_segment(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return _dist_point(p, a)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _dist_polyline(p: Point, points: Sequence[Point]) -> float:
    return min(_dist_segment(p, points[i], points[i + 1]) for i in range(len(points) - 1))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class RegionMapper:
    """Translate normalised minimap positions into behavioural regions."""

    def __init__(self, anchors: Dict[str, object] | None = None) -> None:
        merged = dict(DEFAULT_ANCHORS)
        if anchors:
            merged.update(anchors)
        self.a = merged
        self.blue_base: Point = tuple(self.a["blue_base"])  # type: ignore[assignment]
        self.red_base: Point = tuple(self.a["red_base"])  # type: ignore[assignment]
        self.dragon: Point = tuple(self.a["dragon"])  # type: ignore[assignment]
        self.herald: Point = tuple(self.a["herald"])  # type: ignore[assignment]
        inset = float(self.a["lane_inset"])
        self.top_polyline: List[Point] = [
            (inset, 1.0 - inset),
            (inset, inset),
            (1.0 - inset, inset),
        ]
        self.bot_polyline: List[Point] = [
            (inset, 1.0 - inset),
            (1.0 - inset, 1.0 - inset),
            (1.0 - inset, inset),
        ]

    def classify(self, x: float, y: float) -> str:
        """Return the region label for a normalised minimap position."""
        x, y = _clamp01(x), _clamp01(y)
        point: Point = (x, y)

        if _dist_point(point, self.blue_base) <= self.a["base_radius"] or _dist_point(
            point, self.red_base
        ) <= self.a["base_radius"]:
            return "Base"

        if _dist_point(point, self.dragon) <= self.a["dragon_radius"]:
            return "Dragon Area"

        if _dist_point(point, self.herald) <= self.a["herald_radius"]:
            return "Herald Area"

        river_dist = self.river_distance(point)
        if river_dist <= self.a["river_threshold"]:
            return "River"

        mid_dist = abs(x + y - 1.0) / math.sqrt(2.0)
        if mid_dist <= self.a["mid_threshold"]:
            return "Mid Lane"

        top_dist = _dist_polyline(point, self.top_polyline)
        bot_dist = _dist_polyline(point, self.bot_polyline)
        if top_dist <= self.a["lane_threshold"] and top_dist <= bot_dist:
            return "Top Lane"
        if bot_dist <= self.a["lane_threshold"]:
            return "Bot Lane"

        return "Red Jungle" if x > y else "Blue Jungle"

    def river_distance(self, point: Point) -> float:
        """Distance to the *in-play* segment of the river anti-diagonal."""
        start = float(self.a["river_start"])
        end = float(self.a["river_end"])
        return _dist_segment(point, (start, start), (end, end))

    def forward_direction(self, own_base: str) -> Point:
        """Unit vector pointing from the player's base toward the enemy base."""
        origin = self.blue_base if own_base.lower().startswith("blue") else self.red_base
        target = self.red_base if own_base.lower().startswith("blue") else self.blue_base
        dx, dy = target[0] - origin[0], target[1] - origin[1]
        norm = math.hypot(dx, dy) or 1.0
        return (dx / norm, dy / norm)

    def forward_bias(self, x: float, y: float, own_base: str) -> float:
        """Signed advance toward the enemy in [-1, 1].

        ``0`` sits exactly on the mid-line diagonal, ``-1`` at the player's own
        fountain and ``+1`` at the enemy fountain, so positive values always
        mean *past mid, towards the enemy*.
        """
        direction = self.forward_direction(own_base)
        origin = self.blue_base if own_base.lower().startswith("blue") else self.red_base
        target = self.red_base if own_base.lower().startswith("blue") else self.blue_base
        span = math.hypot(target[0] - origin[0], target[1] - origin[1]) or 1.0
        rel = (x - origin[0], y - origin[1])
        along = (rel[0] * direction[0] + rel[1] * direction[1]) / span
        return max(-1.0, min(1.0, 2.0 * along - 1.0))

    def tower_distances(self, x: float, y: float) -> Dict[str, float]:
        """Distance from ``(x, y)`` to each approximate outer-turret anchor."""
        towers: Dict[str, Point] = dict(self.a["towers"])  # type: ignore[arg-type]
        return {name: _dist_point((x, y), pos) for name, pos in towers.items()}

    def nearest_tower(self, x: float, y: float) -> Tuple[str, float]:
        """Return ``(tower_name, distance)`` for the closest outer turret."""
        distances = self.tower_distances(x, y)
        name = min(distances, key=distances.get)  # type: ignore[arg-type]
        return name, distances[name]

    def is_enemy_side(self, x: float, y: float, own_base: str) -> bool:
        """True when the position falls in the half of the map owned by the enemy."""
        if own_base.lower().startswith("blue"):
            return x > y
        return y > x

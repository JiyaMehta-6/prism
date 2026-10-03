# PRISM - Technical Documentation

This document describes the algorithms, thresholds and formulas used by PRISM.
Every value quoted is the shipped default and can be changed in code or, where
marked, in the Settings dialog.

---

## 1. Video ingestion

### 1.1 Validation (`core/video_loader.py`)

Accepted containers: `.mp4 .mkv .avi .mov` (case-insensitive).
OpenCV probes FPS, frame count, resolution; a file is rejected when:

- the extension is unsupported,
- `VideoCapture` cannot open it (corrupted),
- FPS/frame-count/resolution are all zero and no frame can be decoded.

A decodable file whose duration cannot be computed (`duration == 0`) is
accepted as *unknown duration*: the analysis window falls back to
`early_game_minutes` instead of rejecting the file.

Failures return `VideoInfo(valid=False, error=...)` instead of raising.

### 1.2 Sampling (`core/frame_extractor.py`)

```text
stride = round(native_fps / sample_fps)          # default sample_fps = 1.0
for each frame:
    grab()        # advance decoder without decoding colour planes
    if index % stride == 0:
        retrieve() -> yield (index, index/native_fps, frame)
```

Memory stays constant (one frame at a time) regardless of video length. The
analysis window stops at `early_game_minutes` (default 15 min).

**Hardware decode (benchmark-gated)** - on the first video of a session the
FFmpeg backend is asked to decode with D3D11 hardware acceleration
(`CAP_PROP_HW_ACCELERATION`). 120 frames are timed in hardware and in software
for the same file; hardware is kept only when it is `>= 1.25x` faster,
otherwise (unsupported build, slow driver, odd codec) software is used for the
rest of the session - a session makes the decision exactly once. Per-file
hardware opens are re-verified (5 frames) so one non-decodable file cannot
break the batch. Set `PRISM_HW_DECODE=0` to skip the benchmark and force
software decoding.

---

## 2. Minimap localisation (`vision/roi_detector.py`)

Candidate squares are anchored to the bottom-left and bottom-right corners at
side lengths `0.09 / 0.12 / 0.15 / 0.18` of the frame width (margin `0.004`).
Each candidate is scored in `[0, 1]` on a 64x64 downscale:

```text
score = 0.60 * terrain_ratio        # HSV gate: green (25-75), brown (0-20|160-180),
       + 0.20 * mean_saturation     #                blue (85-135), S>35-40, V>30
       + 0.15 * min(edge_ratio*6, 1)
       + 0.05 * (1 - dark_penalty)
```

The winner must score `>= 0.18`; it is then refined by taking the bounding box
of terrain-coloured pixels (removes the HUD frame). Results are cached per
resolution. Overrides: *Minimap screen side* (`auto|left|right`) and *ROI*
(`x,y,w,h` as 0-1 fractions).

## 3. Player-marker tracking (`vision/minimap_tracker.py`)

Each minimap crop is resized to 160x160 and processed with three cues:

1. **Motion** - background `B` is an occlusion-guarded EMA:

   ```text
   diff = |gray - B|
   thresh = max(10, percentile(diff, 92) * 0.6)
   motion = diff > thresh                     (2x2 opening)
   static = (motion == 0) AND (appearance == 0)
   α = 0.04·static + 0.06·(1 - static)        (foreground adapts faster)
   B <- (1 - α)·B + α·gray
   ```

   The first `5` frames use a fast wash-in (`α = 0.25`) so the seed frame's
   transient content never becomes permanent background.

2. **Appearance** - bright marker gate `V >= 165 AND S <= 115`, unioned with
   team-coloured pixels (blue hue 95-130 / red hue 0-10 & 170-180, S>=120)
   that are also moving. Optional custom HSV ranges can be supplied.

3. **Tracking prior** - an alpha-beta style prediction
   `p_pred = p_prev + v·decay` with decay `0.55` (shrinking while missing).

**Candidate scoring** (connected components, area 3 .. 6% of the crop):

```text
score = (0.34·appearance_ratio + 0.30·motion_ratio + 0.16·brightness
         + 0.20·size_prior) · aspect_penalty · border_penalty
combined = score · (0.72 + 0.35·exp(-d(p, p_pred)² / (2·0.16²)))
```

**Confidence:**

```text
conf = clamp((score - 0.16) / 0.42, 0, 1)   # x0.55 if on the crop border
```

A sample is kept only when `conf >= marker_min_confidence` (default 0.30);
otherwise it is a *miss*. Velocity resets after 3 consecutive misses.

**Interpolation** - gaps of up to `max_interpolate_gap` samples (default 4)
between two observations are filled linearly; interpolated samples carry
`detected=False` and 75% of the neighbouring confidence.

## 4. Region mapping (`vision/region_mapper.py`)

Coordinates are normalised to `[0,1]²` (image convention: x right, y down) with
blue fountain at `(0.05, 0.95)` and red fountain at `(0.95, 0.05)`.

Classification order (first match wins):

| # | Region | Test |
| --- | --- | --- |
| 1 | Base | distance to either fountain `<= 0.10` |
| 2 | Dragon Area | distance to `(0.66, 0.70) <= 0.075` |
| 3 | Herald Area | distance to `(0.34, 0.28) <= 0.075` |
| 4 | River | distance to segment `(0.20,0.20)-(0.80,0.80)` `<= 0.05` |
| 5 | Mid Lane | `|x + y - 1| / sqrt(2) <= 0.065` |
| 6 | Top Lane | distance to polyline `(i,1-i)-(i,i)-(1-i,i)` `<= 0.075`, `i = 0.07` |
| 7 | Bot Lane | distance to polyline `(i,1-i)-(1-i,1-i)-(1-i,i)` `<= 0.075` |
| 8 | Red / Blue Jungle | residual: `x > y` -> Red, else Blue (river halves the map) |

**Own base inference** - among samples classified `Base`, whichever fountain is
visited more often is taken as the player's own side (default Blue).

**Forward bias** - progress along the blue->red axis, remapped so that the own
fountain is `-1`, the mid-line crossing is `0` and the enemy fountain is `+1`:

```text
bias(x, y) = clamp( 2 · ((p - origin)·d / |red - blue|) - 1 , -1, 1 )
```

`forward_high_fraction = share of samples with bias > 0.35`
(`> 0` already means past mid; 0.35 adds a safety margin).

**Tower anchors** (approximate outer turrets, radius 0.055):
blue `(0.09,0.56) (0.30,0.70) (0.44,0.91)`,
red `(0.56,0.09) (0.70,0.30) (0.91,0.44)`.

## 5. OCR & vision identification (`vision/*`)

```text
ROI candidates -> preprocessing variants -> EasyOCR (allowlist) -> confidence
               -> accept if >= ocr_min_confidence (0.55), else keep best attempt
```

Variants: grayscale+upscale(3x)+denoise+contrast, CLAHE, Otsu threshold,
sharpened, inverted. Early exit at confidence `>= 0.90`.

ROIs (fractions of the frame): match clock top-centre (primary, with
top-right/tight fallbacks), blue/red team-kill score boxes beside the clock.

**Clock alignment** - every `ocr_interval_sec` (default 30 s, max
`ocr_max_calls_per_video` = 40) the timer is read; only readings that parse as
a clock are considered (a confident unusable string from one ROI cannot shadow
a parseable reading from another). Offsets
`clock - video_time` are median-filtered; if the spread exceeds 300 s the
clock is declared unstable and video time is used. Rejected keys are reported
by `ConfidenceManager.note_failures()`.

**Parsing** accepts `MM:SS` and `H:MM:SS` only (`vision.ocr_engine.parse_clock`).

**Scoreboard kill events** - the blue/red kill boxes are read on the same
cadence; a positive delta in either box becomes a `kill` event stamped at
video time + clock offset. A decrease or a gap `> 180 s` re-bases the stored
scores instead of emitting a negative kill.

**Champion identity** (`vision/champion_identifier.py`) - at fixed video times
(10/50/100 s) the HUD portrait box (normalised `(0.0016, 0.7935, 0.0359,
0.0630)`, inner 10% inset) is cropped, and a 64-bit pHash (numpy DCT-II) is
compared against the local Data Dragon icon set in `data/champions/`.
`conf = base · (0.5 + 0.5 · min(1, gap/4))` with a logistic
`base = 1/(1 + e^((d-17)/3.5))` over Hamming distance `d`; the identity is
claimed only at `conf >= 0.65` and the majority vote across samples decides.
No match is ever asserted from the network - icons are prefetched once
(`PRISM_ALLOW_DOWNLOAD=1` opt-in), otherwise identification stays disabled.

## 6. Roam detection (`core/behavior_engine.py`)

1. **Primary lane** - the lane holding `>= 10%` of samples with the largest
   share. If no lane qualifies (jungler/support), the fallback detector fires:
   excursions into the enemy half of the map.
2. **Excursion** - consecutive samples outside the primary lane (Base visits
   end an excursion). It becomes a roam when it lasts `>= roam_min_duration_sec`
   (8 s) and is cut off at `roam_max_lane_absence_sec` (45 s).
3. A roam records start/end, origin, most frequent destination region and the
   collapsed path (e.g. `Mid Lane -> River -> Top Lane`).

Derived: `roam_rate = count / analysed_minutes`, destination and route
histograms, median duration.

**Ward (vision bloom) events** (`vision/ward_detector.py`) - after a 10 s
warm-up, frames are compared against an exponential background model
(alpha 0.03); a candidate blob must survive every gate: brightness gain
`> 58`, blue channel `>= 140`, blue-red dominance `>= 30`, area
`0.02%-0.11%` of the minimap, inside the 4% border rejection, and stationary
for `>= 10 s` (moving blobs, portraits and recall swirls fail persistence).
Confirmed blooms are emitted as `ward` events only inside vision corridors
(River, Blue/Red Jungle, Dragon/Herald pits - never lanes or Base) with a
spatial cooldown (same spot within 45 s within 0.12 of the map collapses to
one event), feeding the Vision axis as `ward_events`.

## 7. Per-video metrics

| Metric | Definition |
| --- | --- |
| `region_*` | share of analysed seconds per region |
| `lane/jungle/river/base_fraction` | grouped region shares |
| `objective_fraction` | Dragon + Herald share |
| `vision_proxy_fraction` | `river + 0.5·jungle` share (ward-corridor proxy) |
| `enemy_territory_fraction` | share of samples on the enemy half (`x>y` for blue) |
| `forward_high_fraction` | share with forward bias `> 0.35` |
| `speed_mean`, `speed_p90` | map-units per second between consecutive samples |
| `burst_fraction` | share of speeds `>= 0.9 · p90` |
| `erratic_turn_rate` | heading changes `> 115°` per movement step |
| `tower_presence_fraction` | share of samples within `0.055` of an outer turret |
| `window_forward_std` | std of per-phase mean forward bias (5 phases) |
| `window_speed_std` | std of per-phase mean speed |
| `window_burst_growth` | relative change of phase p90 burst between first and last phase |
| `late_enemy_fraction` | enemy-side share over the late phases (phases 4-5 of 5, the final 40% of the window) |

## 8. Profile aggregation (`core/profiler.py`)

- Region time, transitions and histograms are summed across videos (histogram
  24x24, log-scaled for rendering).
- Every metric gets a time-weighted cross-video mean (`average_metrics`) and a
  cross-video sample spread (`metric_variability`). Consistency coefficients of
  variation are computed *unweighted* from `per_video_metrics` (mean and std
  from the same unweighted samples) so match-to-match spread is measured
  consistently regardless of video length. The mean used as the CV denominator
  is floored at `0.05` (near-zero means cannot explode the ratio) and each
  key's contribution is clamped at `3.0`.
- Quality notes are attached for detection `< 55%` or a missing match clock.

## 9. Fingerprint (`core/fingerprint_engine.py`)

Each axis: `score = 10 · Σ weight_i · saturate(indicator_i)`.
Saturation is `clamp(value / ceiling, 0, 1)`.

| Axis | Indicators (weight, ceiling) |
| --- | --- |
| **Aggression** | forward share (0.35, 0.45) · enemy territory (0.25, 0.35) · burst share (0.20, 0.30) · roam rate (0.20, 0.60/min) |
| **Roaming** | roam rate (0.45, 0.60/min) · destination spread (0.25, 3) · non-lane share (0.30, 0.70) |
| **Vision** | corridor presence (0.60, 0.45) · river share (0.40, 0.20) |
| **Objectives** | pit presence (0.45, 0.12) · turret proximity (0.30, 0.35) · river share (0.25, 0.18) |
| **Risk** | enemy territory (0.35, 0.35) · late enemy exposure (0.25, 0.40) · burst share (0.20, 0.32) · erratic turns (0.20, 0.40) |
| **Consistency** | `0.6·(1 - cv/0.55) + 0.4·(1 - meanAbsDiff/0.22)` over per-video region shares |
| **Pressure Stability** | `0.35·(1 - drift/0.28) + 0.30·(1 - cv_speed/1.4) + 0.20·(1 - max(0,growth)/1.2) + 0.15·(1 - turns/0.45)` |

**Confidence:**

```text
base   = 0.42 + 0.58·detection_rate
volume = min(1, samples / 350)      videos = min(1, n / 3)
conf   = base · (0.55 + 0.25·volume + 0.20·videos) · quality
quality= 0.92 ^ (number of quality notes)      (floor 0.70)
```

Special caps: Vision `<= 0.68` (it is a proxy), Consistency `0.30` with a
single video. Final range `[0.15, 0.97]`.

## 10. Archetypes (`analytics/archetypes.py`)

| Archetype | Rule (all must hold) |
| --- | --- |
| Aggressive Roamer | Aggression `>= 6.3` AND Roaming `>= 6.3` |
| Objective Controller | Objectives `>= 6.5` |
| Lane Dominator | lane share `>= 0.55` AND Roaming `<= 4.8` |
| Safe Farmer | Risk `<= 4.2` AND Aggression `<= 5.2` |
| Team-Oriented Strategist | Objectives `>= 5.8` AND Consistency `>= 6.0` AND Aggression `<= 6.8` |
| Vision-Focused Player | Vision `>= 6.3` |
| High-Risk Playmaker | Risk `>= 6.8` AND Aggression `>= 6.2` |
| Balanced All-Rounder | fallback when no rule fires |

Confidence:

```text
strength   = mean normalised margin past the thresholds
confidence = (0.45 + 0.30·strength + 0.15·evidence + 0.10·quality)
             · (0.75 + 0.25·(0.5·videos + 0.5·quality))   ∈ [0.35, 0.96]
```

Every archetype always ships with the measured reasons that triggered it.

## 11. Pressure episodes (`analytics/pressure.py`)

```text
speed(t)       = distance between consecutive samples / dt
burst_threshold= 85th percentile speed of the recording
stressed(t)    = speed >= threshold AND region in {River, Dragon, Herald} or enemy side
episode        = >= 2 consecutive stressed samples
recovery       = following 3-5 samples show speed std < 0.05
```

`recovery_ratio = recoveries / episodes`.

## 12. Consistency (`analytics/consistency.py`)

- Metric dispersion: mean coefficient of variation of
  `forward_high_fraction, enemy_territory_fraction, lane_fraction, river_fraction,
  roam_rate_per_min` across videos (the same `CONSISTENCY_METRICS` tuple drives
  the fingerprint axis, so both reported numbers always agree).
- Regional overlap: mean pairwise **Jensen-Shannon similarity**
  `1 - JS(P, Q)` of per-video region distributions. Only the worst pair is
  reported when it differs from the best pair.

## 13. Similarity (`analytics/similarity.py`)

Behavioural vector = 7 fingerprint axes/10 + 9 region shares + 8 movement
indicators, L2-normalised; cosine similarity against stored snapshots in
`data/profiles/`. Interpretation bands: `>=0.85` near-identical, `>=0.75` very
similar, `>=0.65` comparable, `>=0.50` loosely related, else distinct.

Snapshots are saved **after** comparison, so a run never matches itself.
Comparison keeps only the strongest row per stored label and skips empty or
non-finite vectors.

**Benchmark percentiles** (`analytics/benchmark.py`) - for each of the 7
fingerprint axes, this run's score is ranked against stored snapshots of
**other** players (`data/profiles/`, same label excluded); the mid-rank
percentile is `100 · (below + 0.5·equal) / n`. The section only appears when
`n >= 3` and every snapshot is complete; snapshots missing any fingerprint
metric are skipped. It surfaces as the `Benchmark` report section (Overview
tab and the PDF generic-sections table) and never blocks a run.

## 14. Timeline (`analytics/timeline.py`)

Windows from `window_plan` (default `0, 3, 7, 12, 15` minutes). Per window:

```text
aggression_index = 10 · (0.45·(bias+1)/2 + 0.35·forward_share(bias>0.3) + 0.20·min(1, roams_per_video/3))
```

Label priority: `Insufficient Data (< 6 samples)` -> `Frequent Roaming (roams
per video >= 2 AND lane share < 0.90)`
-> `Objective Preparation (pits >= 0.12 or minute >= 7 and river >= 0.18)`
-> `Aggressive Trading (index >= 6.0)` -> `Conservative Laning (lane >= 0.55
and index <= 4.5)` -> `Transitional Play`.

Confidence: `0.30 + 0.65·min(1, samples/60)` (+0.08 roams, +0.05 objectives,
-0.08 fallback), capped at 0.93 - thin segments start near 0.37 so a handful
of samples can never masquerade as a conclusion.

## 15. Insights (`analytics/insights.py`)

Confidence formula:

```text
conf = clamp(base · (0.6 + 0.4·fingerprint_conf) · quality, 0.30, 0.96)
quality = clamp(detection_rate, 0.55, 1.0)
```

| Finding | Trigger (defaults) | Suggestion theme |
| --- | --- | --- |
| Frequent overextension during roams | enemy share `>= 0.28` AND roam rate `>= 0.25/min` | push before roaming |
| Low objective participation | Objectives `< 5.5` | arrive ~30 s early, set river control |
| Limited vision corridors | Vision `< 5.5` | ward on rotations/recalls |
| Positioning/movement tempo destabilises | Pressure Stability `< 5.5` (drift-dependent title) | reset after skirmishes |
| Inconsistent early-game pattern | `< 5.5` and `>= 2` videos | fixed 15-minute plan |
| Under-trading in lane | Aggression `< 4.5` (suppressed by Lane Dominator) | punish CS from level 1 |

Positive findings (consistency/objectives/pressure `>= 7.0`) are reported as
strengths. If nothing triggers, a data-insufficiency insight is emitted so the
report is never empty.

## 16. Charts (`visualizations/charts.py`)

`Agg` backend, 140 dpi PNG in `outputs/charts/`:
`fingerprint_radar`, `position_heatmap` (log-scaled 24x24 histogram over an
annotated map layout), `region_distribution`, `tactical_timeline`,
`transition_matrix`, `consistency_variation`, `profile_similarity`,
`fingerprint_bars` (score + confidence).

## 17. PDF (`reports/pdf_export.py`)

A4, ReportLab Platypus, nine sections: cover, executive summary (top findings +
analytics table + quality notes), fingerprint (radar + table), archetypes,
heatmaps/regions/transitions, timeline, insights, recommendations, similarity,
methodology & limitations. Tables highlight headers, alternate row shading and
wrap long text in Paragraph cells.

## 18. Performance notes

| Stage | Cost driver | Typical (15 min @ 1 fps, 1080p) |
| --- | --- | --- |
| Frame streaming | decoder `grab()` | ~0.1-0.3 s per sampled frame |
| ROI detection | once per resolution (cached) | negligible |
| Marker tracking | 160x160 ops per frame | ~2-5 ms |
| OCR | EasyOCR on small ROI | ~0.5-2 s per call, max 12 per video (disable if slow) |
| Analytics + charts | in-memory | < 1 s |
| PDF | vector + PNG embedding | ~1 s |

Parallelism: videos are probed sequentially (invalid files are skipped
before any pool starts), then analysed concurrently in a bounded pool
(`min(videos, cpu, 4)` workers, one `BehaviorEngine` per task, results
re-ordered to the input order). Everything runs off the GUI thread, so the
window stays live; the first video of a session may additionally pay the
one-off hardware-decode benchmark (see section 1.2).

## 19. Logging

`logs/analysis.log` (2 MB x 3 rotations):

```text
2026-01-01 12:00:00 | INFO     | prism.behavior_engine | Stage started: analyze:game.mp4
2026-01-01 12:00:05 | INFO     | prism.behavior_engine | game.mp4: 900 samples (94% detected), 6 roams, own base=Blue
2026-01-01 12:00:06 | WARNING  | prism.confidence | OCR key 'game_timer' rejected (confidence too low, best=0.41 over 3 attempts)
```

Levels: INFO for stages/results, WARNING for degraded inputs, ERROR with
traceback for failures, DEBUG for per-sample detail.

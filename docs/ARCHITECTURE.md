# PRISM - Architecture Documentation

## 1. Design goals

| Goal | How it is achieved |
| --- | --- |
| Explainable intelligence | Pure geometric/statistical models with declared weights; no black-box scoring |
| Offline, free | Local libraries only (OpenCV, EasyOCR, scikit-learn, Matplotlib, ReportLab) |
| Resilient to perception failure | Confidence gating everywhere; missing data degrades confidence, never crashes |
| Responsive GUI | All work happens on `QThread` workers; the UI only reacts to signals |
| Testable | Deterministic units + a synthetic end-to-end pipeline test |
| Maintainable | Layered packages with single-direction dependencies |

## 2. Layered view

```text
┌─────────────────────────────────────────────────────────────┐
│  gui/            PySide6 presentation + orchestration       │
│                  main_window · report_view · settings       │
│                  worker (QThread)                           │
├─────────────────────────────────────────────────────────────┤
│  reports/  pdf_export        visualizations/  charts        │
├─────────────────────────────────────────────────────────────┤
│  analytics/      explainable behavioural insights           │
│                  aggression · roaming · objectives          │
│                  pressure · consistency · similarity        │
│                  archetypes · insights · timeline           │
├─────────────────────────────────────────────────────────────┤
│  core/           pipeline & domain model                    │
│                  behavior_engine · profiler                 │
│                  fingerprint_engine · models · config       │
│                  video_loader · frame_extractor · logger    │
├─────────────────────────────────────────────────────────────┤
│  vision/         perception                                  │
│                  roi_detector · minimap_tracker             │
│                  region_mapper · ocr_engine                 │
│                  ocr_preprocessor · confidence_manager      │
└─────────────────────────────────────────────────────────────┘
```

Dependency rule: **upper layers may import lower layers, never the reverse.**
`vision` never imports `analytics`; `analytics` never imports `gui`.

## 3. Module responsibilities

### gui/
| File | Responsibility |
| --- | --- |
| `main_window.py` | Window layout, video list (add/remove/drag-drop), actions, status/log/progress, stylesheet |
| `worker.py` | `AnalysisWorker` (probe -> optional per-video thread pool -> aggregate), `PdfExportWorker`, `ClipExportWorker` as `QThread`s with signals |
| `report_view.py` | Seven report tabs rendering an `AnalysisReport` |
| `settings_dialog.py` | Bounded, validated editor for `core.config.Settings` |

### core/
| File | Responsibility |
| --- | --- |
| `models.py` | All dataclasses (`VideoInfo`, `PositionSample`, `RoamEvent`, `PlayerProfile`, `Fingerprint`, `Archetype`, `Insight`, `TimelineWindow`, `SimilarityResult`, `AnalysisReport`) + JSON helpers |
| `config.py` | Settings schema, defaults, validation, directory layout |
| `logger.py` | Rotating file + console logging, `StageTimer` performance markers |
| `video_loader.py` | Format allow-list, OpenCV probing, never-raising validation |
| `frame_extractor.py` | Streamed sampling at N fps using `grab()`/`retrieve()` (constant memory); session-cached, benchmark-gated hardware decode |
| `behavior_engine.py` | Per-video pipeline: track -> interpolate -> regions -> clock -> roams -> deaths + awareness micro-metrics -> metrics |
| `profiler.py` | Merge N video analyses into one `PlayerProfile` (region time, transitions, variability) |
| `fingerprint_engine.py` | Seven-axis 0-10 fingerprint with weights, confidence and evidence strings |

### vision/
| File | Responsibility |
| --- | --- |
| `roi_detector.py` | Locates the minimap square (either corner) via a map-likeness score, refines to terrain bounds, caches |
| `minimap_tracker.py` | Marker detection: background-motion + appearance gate + tracking prior -> `(x, y, confidence)` |
| `region_mapper.py` | Normalised coordinates -> region labels, forward bias, tower/base distances |
| `ocr_preprocessor.py` | Grayscale/upscale/denoise/CLAHE/threshold/sharpen variants |
| `ocr_engine.py` | Lazy EasyOCR wrapper, ROI selection, allowlists, best-variant selection, clock parsing; reader guarded by a re-entrant lock for parallel videos |
| `ward_detector.py` | Background-motion blob gate -> stationary blue-spirit "vision blooms" in corridor regions (ward events) |
| `champion_identifier.py` | HUD portrait crop -> 64-bit pHash vs local Data Dragon icons, confidence claim gate |
| `confidence_manager.py` | Per-key acceptance gating, best-reading retention, failure reporting |

### analytics/
| File | Output |
| --- | --- |
| `aggression.py` | `AggressionAnalysis` (forward frequency, engagement tendency, risk exposure, roam frequency) |
| `roaming.py` | `RoamingAnalysis` (rate, duration, destinations, routes, timing buckets) |
| `objectives.py` | `ObjectivesAnalysis` (pit presence, timing buckets, turret proximity, objective rotations) |
| `deaths.py` | Vision-only death inference: marker dropout gaps that end at the own fountain, scoreboard-confirmed (`DeathInference`) |
| `awareness.py` | Micro-metrics: unwarded forward share (vision-adjusted risk) and reaction latency after scoreboard kills |
| `pressure.py` | `PressureAnalysis` (drift, burst growth, stress episodes, recovery ratio) |
| `consistency.py` | `ConsistencyAnalysis` (metric dispersion, Jensen-Shannon regional overlap, pairwise table) |
| `similarity.py` | Behavioural vectors, snapshot persistence, cosine similarity ranking |
| `benchmark.py` | Per-axis mid-rank percentiles vs other stored profiles (needs `n >= 3`, excludes the analysed label) |
| `archetypes.py` | Rule-based classification with margin-based confidence and reasons |
| `insights.py` | Confidence-scored findings + suggestions + evidence |
| `timeline.py` | Segmented match windows (Early/Mid/Late/End phases) with labels and confidence |

### reports/ & visualizations/
| File | Responsibility |
| --- | --- |
| `pdf_export.py` | Nine-section ReportLab document; missing charts are skipped, not fatal |
| `clips.py` | Short re-encoded MP4 clips around each detected roam (OpenCV writer, fully offline) |
| `charts.py` | Headless Agg charts saved to `outputs/charts/` (radar, regions, timeline, transitions, consistency, similarity, bars) |

## 4. Data flow

```text
video files
   │  probe_video()                         core/video_loader
   ▼
VideoInfo ──► BehaviorEngine.analyze()      core/behavior_engine
                 │
                 ├─ ROIDetector.detect()        vision/roi_detector
                 ├─ FrameExtractor.iter_frames()  core/frame_extractor  (streamed)
                 ├─ MinimapTracker.process()      vision/minimap_tracker  → (x, y, conf)
                 ├─ OCREngine.read_game_clock()   vision/ocr_engine      → clock seconds
                  ├─ interpolate + window filter
                  ├─ RegionMapper.classify()       vision/region_mapper   → region labels
                  ├─ roam detection
                  ├─ detect_deaths()               analytics/deaths       → death events
                  ├─ awareness micro-metrics       analytics/awareness    → unwarded / reaction
                  └─ metrics
   ▼
VideoAnalysis (per video)
   │  build_profile()                      core/profiler
   ▼
PlayerProfile ──► FingerprintEngine.compute()   core/fingerprint_engine
   │                     │
   │                     ▼
   │               Fingerprint (7 axes + confidence + evidence)
   │                     │
   ├─────────────────────┼──────────────────────────────────┐
   ▼                     ▼                                  ▼
analytics sections    classify_archetypes()            generate_insights()
   │                     │                                  │
   └──────► build_timeline() ◄──────────────────────────────┘
                     │
                     ▼
   charts.generate_all()  +  compare_profiles()  +  compute_benchmark()
                      │
                      ▼
                AnalysisReport ──► ReportView (GUI)
                                └─► export_pdf() → player_report.pdf
                                └─► export_clips() → outputs/clips/*.mp4
                                └─► save_json()  → last_analysis.json
                                └─► save_profile_snapshot() → data/profiles/
```

## 5. Threading model

```text
Qt main thread                     AnalysisWorker (QThread)
─────────────                      ─────────────────────────
render UI, handle clicks           probe videos (sequential, skip invalid)
paint progress bar  ◄── progress(int)
show stage text     ◄── stage(str)     │ valid videos
append log lines    ◄── log(str)       ▼
show report tabs    ◄── finished_report(object)   ThreadPoolExecutor
enable export                              │  ≤ min(videos, cpu, 4) workers,
                                           │  one BehaviorEngine per task,
                                           │  results re-ordered by input index
                                   PdfExportWorker (QThread)
show path / open    ◄── finished_path(str)
                                   ClipExportWorker (QThread)
show paths / open   ◄── finished_paths(list)
```

Rules:

- Workers never touch widgets; they only emit signals.
- `cancel()` sets a cooperative flag checked between videos; in-flight pool
  tasks are dropped with `shutdown(cancel_futures=True)`.
- Each pool task owns its `BehaviorEngine`, so cancellation flags and stride
  state never race between videos; the OCR reader init/read is lock-guarded.
- `closeEvent` cancels and waits for the worker (max 3 s).
- Matplotlib uses the non-interactive `Agg` backend, so charts are safe to
  render off the GUI thread.

## 6. Error-handling strategy

| Failure | Behaviour |
| --- | --- |
| Missing / unsupported / corrupted video | `VideoInfo.valid=False` + reason; batch continues with remaining files |
| Minimap not detectable | Layout fallback ROI, warning logged, low detection rate recorded |
| Marker never found | Empty sample list; profile still produced, quality note added |
| OCR import/initialisation failure | OCR disabled for the session, analysis continues |
| Low-confidence OCR read | Rejected by `ConfidenceManager`; video time used; report notes it |
| Chart rendering error | That chart is skipped, others still rendered, PDF omits it |
| No valid video at all | Worker emits `failed(str)`; GUI shows a dialog, never crashes |
| Unexpected exception | Logged with traceback to `logs/analysis.log`, surfaced once |

Every stage boundary is wrapped by `StageTimer`, so timing and failures are
visible in the log.

## 7. Persistence

| Path | Content | When |
| --- | --- | --- |
| `data/settings.json` | user settings | on Settings → OK |
| `data/profiles/*.json` | behavioural vector snapshots | after each successful analysis |
| `outputs/last_analysis.json` | full machine-readable report | after each successful analysis |
| `outputs/charts/*.png` | every chart | during report rendering |
| `outputs/*_report.pdf` | PDF report | on Export PDF |
| `outputs/clips/*.mp4` | roam clips (mp4v re-encode) | on Export Roam Clips |
| `logs/analysis.log` | rotating log (2 MB x 3) | always |

## 8. Extension points

- **New archetype**: add a predicate to `analytics.archetypes.RULES`.
- **New insight**: add a rule function in `analytics.insights.generate_insights`.
- **New fingerprint axis**: extend `FINGERPRINT_METRICS`, add a composition in
  `fingerprint_engine.COMPOSITIONS`, add radar/timeline rendering.
- **Different map geometry**: override anchors via `vision.region_mapper.DEFAULT_ANCHORS`.
- **New chart**: add a generator to `visualizations.charts.generate_all`.
- **Extra OCR targets**: register ROIs in `vision.ocr_engine.DEFAULT_ROIS`.

## 9. Testing strategy

- `tests/test_units.py` - deterministic units: format validation, probing,
  sampling rate, region geometry, clock parsing, confidence gating,
  fingerprint ranges, settings round-trip.
- `tests/test_pipeline.py` - end-to-end on a synthetic recording with a known
  marker route: tracking quality, region variety, profile, fingerprint,
  analytics, archetypes, insights, timeline, charts, similarity, PDF.
- `tests/video_synth.py` - deterministic synthetic gameplay generator.

Run with `python -m pytest tests -q`.

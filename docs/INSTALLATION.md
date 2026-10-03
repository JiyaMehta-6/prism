# PRISM - Installation Guide

Target platform: **Windows 10 / Windows 11**, Python **3.12**, package manager
**uv**. All dependencies are free and open source; PRISM runs fully offline
after installation.

---

## 1. Prerequisites

| Requirement | Check / install |
| --- | --- |
| Python 3.12 | `py -0p` (uv can also download it automatically) |
| uv | `winget install astral-sh.uv` then reopen the terminal |
| ~4 GB disk | PyTorch (CPU) + EasyOCR models |
| 16-32 GB RAM recommended | analysis itself streams frames and stays flat |

Optional: a dedicated GPU. **Not required** - EasyOCR is forced to CPU
(`gpu=False`) and the vision pipeline is pure OpenCV/NumPy.

## 2. Create the environment

```powershell
cd path\to\PRISM
uv venv --python 3.12
```

If Python 3.12 is not installed, uv downloads a managed build automatically.

## 3. Install dependencies

```powershell
uv pip install -r requirements.txt
```

Installed packages:

```text
PySide6          native desktop GUI
opencv-python    video decoding, frame processing, preprocessing
numpy            numeric core
easyocr + torch  offline OCR (match clock only)
scikit-learn     similarity / vector utilities
matplotlib       charts
reportlab        PDF export
pillow, scipy    image and numeric helpers
```

## 4. Make OCR work offline (one-time, while online)

EasyOCR keeps its recognition models in `%USERPROFILE%\.EasyOCR\model`.
**PRISM never downloads them by default** - if they are missing, OCR is
simply switched off with a warning and everything else still works. To enable
OCR, prefetch the models once during installation:

```powershell
.venv\Scripts\python.exe -c "import easyocr; easyocr.Reader(['en'], gpu=False)"
```

Alternatively, allow an explicit one-off download at launch with the
`PRISM_ALLOW_DOWNLOAD=1` environment variable. Without either step you can
still use PRISM - switch **OCR off** in *Settings*; the whole behavioural
pipeline works without OCR.

## 5. Verify the installation

```powershell
# unit + end-to-end tests (generate a synthetic video, run the full pipeline)
uv pip install pytest
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: `21 passed` (includes a hermetic test that blocks all sockets while
running the pipeline).

Optional lint check:

```powershell
uv pip install ruff
.venv\Scripts\python.exe -m ruff check . --line-length 110
```

## 6. Launch

```powershell
.venv\Scripts\python.exe main.py
```

Create a shortcut for daily use:

```powershell
$exe = (Resolve-Path .venv\Scripts\python.exe).Path
$main = (Resolve-Path main.py).Path
$s = (New-Object -ComObject WScript.Shell).CreateShortcut("$env:USERPROFILE\Desktop\PRISM.lnk")
$s.TargetPath = $exe
$s.Arguments = "`"$main`""
$s.WorkingDirectory = (Get-Location).Path
$s.Save()
```

## 7. First run checklist

1. Add 1-5 gameplay videos (MP4 / MKV / AVI / MOV).
2. Open **Settings** and confirm:
   - Frame sampling rate `1.0 fps` (default),
   - Max game time `45 min` (whole match, phases Early / Mid / Late / End),
   - Minimap side `auto` (or `left`/`right` if you know your HUD),
   - OCR enabled only if the models were prefetched.
3. Click **Analyse Player**.
4. Watch the green progress bar, the stage label and the log pane.
5. Review the tabs, then **Export PDF**. Optionally **Export Roam Clips**
   to watch each roam as a short MP4.

Outputs:

```text
outputs/player_report.pdf      PDF report
outputs/last_analysis.json     machine-readable export
outputs/charts/*.png           every rendered chart
outputs/clips/*.mp4            roam clips (OpenCV mp4v, offline)
data/profiles/*.json           profile snapshots used for similarity
logs/analysis.log              rotating log (errors, warnings, timings)
```

## 8. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Unsupported format .xxx` | Only MP4/MKV/AVI/MOV are accepted; re-encode with ffmpeg if needed |
| `Unable to open (corrupted file)` | Container is damaged; re-export the recording |
| `No decodable frames found` | Missing codec pack - re-record or convert to H.264 MP4 |
| Minimap not found / nonsense regions | Set **Minimap screen side** manually, or enter an ROI override `x,y,w,h` as 0-1 fractions of the frame |
| Low marker detection warning | Reduce *Minimum marker confidence*, or use a cleaner 1080p recording; the report still generates with lower confidence |
| OCR always "Value unavailable" | Models not prefetched or timer outside the ROI - disable OCR and rely on video time (stated in the report) |
| Slow analysis | Lower the sampling rate (e.g. `0.5 fps`), disable OCR, analyse shorter clips |
| Very high memory usage | Close other applications; PRISM streams frames and does not load whole videos into RAM |
| GUI looks tiny on 4K | Windows display scaling is honoured via Qt; log out/in after changing scale |

## 9. Uninstall

Delete the project folder and (optionally) `%USERPROFILE%\.EasyOCR`.
No services, registry keys or system-wide packages are created.

## 10. Offline statement

PRISM performs **no network calls** at run time: video decoding, tracking,
OCR, analytics, charts and PDF generation are all local. EasyOCR model
downloads are disabled unless you opt in (`PRISM_ALLOW_DOWNLOAD=1` or the
manual prefetch in step 4), and `tests/test_offline.py` re-runs the pipeline
with every socket blocked to prove it. No API keys, accounts, credit cards,
subscriptions or cloud services are involved - PRISM costs nothing beyond the
hardware it runs on.

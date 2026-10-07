# AGENTS.md — PRISM workspace boundaries

## What this repository is
PRISM: a fully **offline, free** PySide6 desktop app that analyses League of
Legends gameplay videos into an explainable behavioural report (charts + PDF +
roam-clip export). Hard rules: no fees, logins, paid APIs, cloud, telemetry, or
runtime downloads unless gated behind `PRISM_ALLOW_DOWNLOAD=1`.

- Remote: https://github.com/JiyaMehta-6/prism (private), branch `master`
- Toolchain: `.venv` (Python 3.12), `pyproject.toml`, `requirements.txt`
- Runtime data: `data/` (settings, profiles, champion icons, sample videos),
  `logs/`, `outputs/` — all git-ignored except committed champion icons

## Hard boundary
`D:\projects\2026\PRISM` contains **only PRISM**. A sibling, unrelated project
lives at `D:\projects\2026\Clash of Clans` (Clash of Clans "coc-databank").
They share nothing. Specifically:

- Never edit, create, lint, test, or run anything under
  `D:\projects\2026\Clash of Clans` from a session rooted in this repository.
- Never add `CoC` (or any workspace member) to this `pyproject.toml` — no
  `[tool.uv.workspace]`, no `uv.lock` referencing other projects.
- Never install `coc-databank` (or any foreign package) into this `.venv`.
- Never place other projects' files inside this folder tree.

Incident note: a foreign session twice nested `CoC/` inside this repo (first as
a uv-workspace member + editable install, later as plain files + a stray
`uv.lock`/`test_regex.py`). All of it was removed and the CoC project was
relocated to `D:\projects\2026\Clash of Clans` (2026-10-07); git history was
verified clean. Do not reintroduce it.

## Verification (run from this folder)
```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m ruff check . --line-length 110 --no-cache
.venv\Scripts\python.exe -m pytest tests -q          # 75 tests, ~6 min
# GUI smoke (offscreen): C:\Users\Jiya\AppData\Local\Temp\opencode\gui_smoke.py
```

# Visual Intel 2.0 — Engineering Upgrade Plan

Working plan for upgrading visual-intel toward ~9.5/10 engineering quality.
Incremental, phase-by-phase, on branch **`engineering/visual-intel-2.0`**.
One commit per phase, tests after each, honest reporting. **Not a rewrite** —
adapt to the existing structure; preserve working features and interfaces.

## Context for whoever picks this up
- **What it is:** local-first webcam intelligence app (Windows 11, system Python 3.14,
  no venv). One process: camera → YOLO/YOLO-World (GPU via DirectML) → tracker → events
  → VLM (Ollama) → faces (InsightFace, GPU) → diary (JSONL) → OpenCV window or localhost web.
- **Assumed load:** one person, one camera, one browser tab, on their own PC. Judge all
  scale decisions against this — NOT a multi-user server.
- **Repo:** https://github.com/mirzafasahatullabaig-glitch/visual-intel (private).
  `master` is at tag `v0.9.2` (commit `ca3f759`). Work happens on `engineering/visual-intel-2.0`.
- **Push:** the user runs every `git push` (agent push is blocked). Never push.
- **Each module self-checks** when run alone (`python detect.py`, etc.). `test_e2e.py`
  needs a person at the camera. Models (.pt/.onnx), `faces/`, `memory/` are gitignored.
- **Install traps:** never plain `pip install faster-whisper` or anything that pulls plain
  `onnxruntime` over `onnxruntime-directml`. Use `--no-deps`. Faces need insightface+onnx+scikit-image.
  Open-vocab needs ultralytics+CLIP (torch already present, CPU build).

## Scope decisions already made (do NOT reopen without asking)
These were judged as cost > value for a single-camera local app, and the original spec
explicitly allows adapting rather than forcing them:
- ❌ **No `visual_intel/` package migration** — breaks every self-check and run command for
  near-zero reliability gain. Keep the flat module layout.
- ❌ **No full SQLite+FTS5 rewrite of memory** — JSONL + atomic writes is already crash-safe;
  the user wants keyword/synonym recall. Only revisit if the diary genuinely outgrows RAM.
- ❌ **No ByteTrack/BoT-SORT** — a dependency for a problem a desk camera doesn't have.
  The IoU tracker works. (May still add motion prediction / better association in-place.)
- ❌ **No auth** — app is localhost-only by design (binds 127.0.0.1, Host+Origin checks).
  The spec says "keep default local-only," which it already is.
- ❌ **No frame sequence numbers** — the design reads the single newest frame and skips by
  identity; there is no queue to go stale. Already meets "no stale frame processing."

## Progress
- [x] **Phase 0** — baseline + branch `engineering/visual-intel-2.0` created off `ca3f759`.
- [x] **Phase 1 (partial)** — commit `c769ff8`:
      `guarded()` escalating backoff + give-up (fixes faces-off busy-loop and runaway
      restarts); unique snapshot filenames + imwrite-success check in `memory.py`.

## Remaining phases

### Phase 1 (finish) — critical bugs
- **detect.py:** replace import-time model load with explicit `init()`; validate the ONNX
  output shape (`(1, C, 8400)`) and class count vs NAMES; wrap inference so one bad frame
  doesn't kill the loop; make detection cadence configurable (from config.py).
- **track.py:** verify expired tracks are removed BEFORE matching new detections (check the
  current order in `update()`); ensure an expired ID is never reused; make `visible()` 0.3s
  window and `FORGET`/`CONFIRM` configurable.
- **main.py:** graceful shutdown — store thread refs in `start_workers`, join them in the
  `finally` of `main()`; confirm `S["running"]=False` actually stops every loop. Camera
  backend already handles Windows (CAP_DSHOW); add a Linux/Pi fallback (default backend).

### Phase 2 — config + lifecycle (the backbone of "configurable")
- **config.py (new):** one module of tunables with hardware profiles (desktop-gpu / cpu / pi):
  detector cadence, CONF/NMS, face EVERY/THRESHOLD/det_size, memory MAX_ENTRIES/SNAPSHOT_DAYS,
  VLM model + timeout, web PORT. Read env var `VI_PROFILE` to pick a profile. Every module
  imports its constants from here instead of hardcoding. Keep current values as the gpu default.
- **Service lifecycle:** a light pattern (init/start/health/stop) for detector, faces, camera,
  web, voice — one failing optional service must not crash others (mostly true already via
  `guarded` + optional imports; formalize and test it).

### Phase 3 — perception + reasoning
- **tracker:** optional motion prediction (last velocity) to hold IDs through brief occlusion;
  track confidence. Benchmark against current before keeping. In-place, no new dep.
- **vlm.py:** connection health check (`/api/tags` ping), configurable timeout, response
  validation (done in v0.9.2), request cancellation / bounded request queue to replace the
  single `S["question"]` slot, latency measurement. Keep detection / scene-desc / Q&A separate.
- **events/watch.py:** event IDs, severity, dedup + cooldown (don't re-alert the same thing
  every frame), acknowledgement, persistent event history. Keep deterministic rules separate
  from any LLM judgment.

### Phase 4 — memory (careful, preserve API)
- Keep JSONL as default. Add: record validation on load, configurable retention (have cap +
  prune already). SQLite only as an *optional* backend if the user wants it — not mandatory.
  Preserve `remember()` / `recall()` signatures.

### Phase 5 — faces + privacy
- Validate embedding dimensions (512) and stored records on load. Atomic write to known.json
  (temp + os.replace). Confidence margin (2nd-best gap) to reduce false matches.
- **Controls:** forget one identity, disable recognition at runtime, wipe all biometric data.
- **PRIVACY.md:** what is stored (embeddings + snapshots), where, how to delete, that nothing
  is uploaded. This is a real scoring dimension.

### Phase 6 — web + voice
- **web.py:** already has Host/Origin + size caps + path-traversal guard + None-frame guard.
  Add: per-endpoint input validation pass, non-overlapping polling already done client-side,
  face-naming failure feedback (toast on false return), graceful disconnect.
- **voice.py:** recording cancellation, max duration (have MAX_SECONDS), TTS process
  monitoring, Whisper error handling, configurable language (WHISPER const → config.py),
  mic failure recovery (partly done). Keep optional.

### Phase 7 — performance + metrics
- **monitoring (new, light):** measure camera FPS, detector FPS, frame latency, VLM latency,
  queue depth, CPU/RAM (psutil if available, else skip). Surface on the web page + a `/metrics`
  JSON. No hardcoded perf claims; measure on this machine and record in PERFORMANCE.md.
- Hardware profiles from config.py drive cadence/resolution for cpu / pi.

### Phase 8 — testing + CI (biggest score lever)
- **tests/ (pytest):** fold existing self-check logic into real unit tests —
  tracker association/expiry/no-ID-reuse, memory persistence/retrieval/cap/atomic,
  watch rules/cooldown, faces match/validate/cap/blank-name, bbox decode, guarded backoff,
  config validation, request queue.
- **failure tests:** missing model, corrupt diary line, Ollama down (mock), camera unavailable,
  GPU init failure→CPU fallback, full queue, worker exception, disk write failure, bad API input.
- **hardware tests:** marked `@pytest.mark.hardware`, excluded from normal CI.
- **GitHub Actions:** run pytest (non-hardware) + Ruff + mypy on push. pytest-cov report.

### Phase 9 — security review
- Re-audit: localhost binding, Host/Origin, input validation, path traversal, snapshot access,
  the PowerShell TTS command (text via stdin — already safe), temp files, biometric storage,
  dependency vulns, secret leakage. Document findings. Keep default local-only.

### Phase 10 — docs + release
- README (done-ish), ARCHITECTURE.md, INSTALLATION.md, CONFIGURATION.md, TESTING.md,
  PERFORMANCE.md (measured, not claimed), PRIVACY.md, TROUBLESHOOTING.md, CHANGELOG.md.
- Architecture diagram (ASCII is fine). Example config. Reproducible setup script.
- Final engineering report: modified files, resolved bugs, test evidence, coverage, measured
  perf, security findings, known limits, and a per-dimension score (architecture, code quality,
  reliability, performance, CV, AI integration, memory, security, testing, docs) — only as high
  as the evidence supports.

## Workflow rules (from the spec, keep following)
1. Inspect real code before changing; never trust a stale review.
2. Smallest fix that works; fix root causes; avoid new deps.
3. One commit per phase; run tests; report failures honestly; don't proceed past a critical regression.
4. Never commit secrets, biometric data, or personal snapshots.
5. Ask before destructive changes or altering persistent user data (e.g. a diary format change).
6. After each phase: show modified files, explain why, run tests, report.

## How to resume in a new chat
Say: "continue Visual Intel 2.0 from ENGINEERING_PLAN.md". Then:
1. `cd F:\visual-intel && git checkout engineering/visual-intel-2.0 && git log --oneline -5`
2. Read this file's Progress section for the last finished phase.
3. Re-read the actual files before editing (code may have moved).
4. Continue the next unchecked phase.

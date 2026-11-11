# Security review: Phase 3 completion (evaluation harness, WP4a)

Date: 2026-11-10. Reviewer: Security Reviewer agent. No files modified except this report.

## 1. Scope
Read in full: backend/evaluation/remote_zip.py, dataset.py, cli.py; the security-relevant parts of runner.py, bench.py, perturb.py, report.py (grep for subprocess, pickle/eval/exec, logging, environ). Production diffs: backend/app/services/segmentation/face_parser.py (lines 196-250), backend/app/services/facial_analysis/landmarks.py (105-175). Data: docs/evaluation/*, docs/plans/reports/12, .gitignore, ~/.cache/makeup_demo/. Reproduction scripts were run from the scratchpad (python -I, fake in-memory fetcher, no network). pip-audit was run against requirements.txt (queries a public vulnerability service; no project data sent).

## 2. Findings

### F1 (LOW, non-blocking): corrupt-deflate member raises zlib.error, not FetchError
- Location: backend/evaluation/remote_zip.py `fetch_member`, decompress at ~line 226; the catch in dataset.py `fetch_samples` (`except FetchError`) and cli.py `main` (catches CacheError, FetchError, ValueError, OSError).
- Evidence (ran): a zip whose deflate stream was corrupted, read through `fetch_member` -> exception MRO `zlib.error, Exception`. The 200 MB-of-zeros bomb with a 25 MB cap -> `FetchError: member exceeds the size cap` (the bomb guard works).
- Related: a local header shorter than 30 bytes that still starts with `PK\3\4` (member at the very end of the archive) would raise `struct.error` at `struct.unpack("<HH", raw[26:30])`. Inferred from the code, not run.
- Impact: the CLI exits with a traceback instead of a clean error. Because the exception escapes the `with ThreadPoolExecutor` block without `cancel()`, queued tasks keep downloading until finished or until the byte budget (default 3 GiB) stops them. The budget bounds the damage; there is no memory or code-execution consequence. Requires a hostile or corrupt server (the archives are pinned by Content-Length only).
- Mitigation: wrap decompress and the header unpack in `except (zlib.error, struct.error)` -> FetchError; cancel pending futures on any exception.

### F2 (LOW, non-blocking): absolute home path and username in a plans document
- Location: docs/plans/reports/12_coder_report_phase3_eval.md:42 (`cd /Users/test/Makeup_demo/backend`). It is the only match under docs/ (grep `/Users/test`). Requirement "no hostname/username/home paths" is therefore not fully met in docs/plans.
- Impact: low (local username disclosed if the repo is published).
- Mitigation: replace with `cd backend`.

### F3 (INFORMATIONAL): fetch limits are per-socket, there is no retry logic
- remote_zip.py: `TIMEOUT_S = 60` applies to each socket operation, not to a whole request, so a slow-drip server can hold a request open for a long time, up to the range size. The budget is still reserved before each request, so total bytes are bounded. No retries exist (a failure aborts); this is the safe direction. The plan item "retries" is not implemented, which is a functional matter, not a security one.

### F4 (INFORMATIONAL): CLI output paths are not constrained
- cli.py `sample --out`, `report --out-prefix`, bench/perturb `--out-prefix` accept arbitrary paths and overwrite. This is a local-operator tool, so it is not a vulnerability. Output of `report` is aggregate-only by construction.

### F5 (INFORMATIONAL): WP4a production changes
- face_parser.py: the `cv2.resize` to an explicit `new_size` is applied to `image[ry0:ry1, rx0:rx1]` and is only a shrink (`shrink < 1.0`), so output is smaller than the input crop, and the input is already capped at 25 MP by Phase 1. Size has a floor of 1. A degenerate region is rejected earlier (`rx1 <= rx0`). No new memory growth, no new crash path observed. Logging stays type-name only (lines 132, 206).
- landmarks.py: a face with fewer than 478 points (including empty) now returns `no_face_detected` with HTTP 200 instead of an IndexError/500. Empty-face area guard added. The log line carries only the point count (line 156). The message is a fixed string. No leak.

## 3. Verified controls (positive)
- Network: https scheme enforced before every request (remote_zip.py `_check_url`); redirects refused via `_NoRedirect` (raises FetchError); 206 required and Content-Range must equal the requested start/end exactly; body read capped at requested+1 bytes and length verified; `Accept-Encoding: identity`; only three pinned URLs accepted (`BudgetedFetcher` rejects unpinned URLs and `end >= pinned`); Content-Length checked by HEAD against pins; byte budget reserved under a lock before the request (no overshoot by threads). The 200-response case aborts without reading the body.
- Zip parsing: central directory parsed by stdlib zipfile over a tail view, tail cap 32 MB, member cap 200,000, encrypted members rejected, only STORED/DEFLATED, size caps (images 25 MB, annotations 2 MB) checked from the header and again during decompression via `decompressobj(max_length)`, size and CRC-32 verified, local-header name compared with the central one. Nothing is extracted by name: zip-slip does not apply; output file names are built only from `require_id` (strict UUID regex fullmatch). Member names come from `IMAGE_MEMBER.format(id=...)` with sample ids that were validated.
- Paths: `refuse_inside_repo` resolves symlinks and refuses the repo, its children and its parents; `clean_cache` requires the marker file, removes only known subdirectories/files, `rmdir`s the root only if empty, and `shutil.rmtree` refuses a symlinked `images/` dir. Atomic writes with mkstemp, mode 0600; dirs 0700.
- Decoding: images go through the production `load_rgb_image` (allow-list, byte and 25 MP caps before decode). Annotations: `Image.open(..., formats=["PNG"])`, mode L/P only, shape checked against the image before `load()`, values < number of classes.
- No pickle/eval/exec/yaml/np.load in the package (grep). Subprocess uses: `sysctl`, `pmset` (fixed argv, no shell, no user input) and bench.py's worker (`[sys.executable, "-m", "evaluation.bench", "--worker", ...]` with `str()` of tuple constants, list form, cwd fixed, timeout 1800). No command injection path. Environment is inherited, nothing secret is passed.
- user_id: read by header name only (`parse_meta` uses `need` columns; `user_id` is never indexed); the sanitized meta CSV in the cache has only image_name,height,width,brightness. grep for `user_id` in docs/evaluation: none.
- Logging: logger calls carry ids, counts, timings, exception type names only (grep over evaluation/).
- pip-audit 2.10.1 on requirements.txt: "No known vulnerabilities found". requirements.txt (mtime 17:13, before the work started at about 19:10) lists pip-audit as a dev tool, and the package has no third-party imports beyond numpy, Pillow, cv2 (already pinned). No new dependency observed (no git history to diff, so this is based on the mtime and the import list).

## 4. Data protection
- docs/evaluation/: 8 files; the sample file has 600 lines that all match `uuid<TAB>dev|test<TAB>q<TAB>g` plus comments (checked by regex; no other lines). All 50 distinct UUIDs in the JSON report are in the sample (0 outside). No `/Users/`, no hostname, no user_id. The report discloses `cpu: Apple M2 Pro` and the macOS version: hardware class only, acceptable. No images or pixels.
- .gitignore: `.cache/`, `easyportrait*/`, the marker, `per_image.jsonl`, `run_meta.json`, image/zip extensions under docs/evaluation. Adequate as a second line. (The repo is not under git here, so this cannot be exercised.)
- ~/.cache/makeup_demo/: total 1.2 GB. easyportrait/ 0700, files 0600 (`find -perm +077` over files returned nothing), no symlinks, marker present, images/ and annotations/ 0700 with 600 + 600 files, manifest.json 170 KB, meta_test.csv 392 KB. Run directories 0700, per_image.jsonl and run_meta.json 0600, robustness/perturb_rows.jsonl 0600. The parent ~/.cache/makeup_demo and easyportrait_runs are 0755 (they hold only 0700 directories). Image content was not read.
- Incident (curl, about 1.38 GB outside the harness): I searched the home directory and /tmp, /var/folders and the session scratchpads for archives >50 MB. The only large files are unrelated user Downloads and test `.onnx` files; no test.zip/annotations.zip or partial downloads remain. Confirmed gone. Could the harness do the same? No: the ranged GET aborts on any non-206 status (a 200 is not read), Content-Range must match, the body read is capped at requested+1 bytes, and total bytes are capped by the budget. The curl case (`-r -4000000-`, malformed) is the failure the harness explicitly guards against. Residual: the coder's own ad-hoc commands bypass the guards; this is a process issue, not a code one (recommend: use the harness or HEAD-only for layout inspection).

## 5. Licence compliance
Attribution present: docs/evaluation/easyportrait_test_2026-11-10.md lines 333-338 (authors, repository URL, licence name and link, BibTeX), docs/face_segmentation.md:167-168, sample file header. The licence is described as a reworked CC BY-SA variant, not CC, with no non-commercial clause and no portrait/privacy rights granted. No images, annotations or per-image rows are in the repo; only UUIDs and aggregates. Use is evaluation only. The statement "none are used for training" is respected by the code (no training paths). Note: UUID lists plus per-image worst-case UUID tables are identifiers, not content, and fall under the "only UUIDs and aggregates" rule. Acceptance criterion 7 (no image/annotation/per-image/pixel data in the repo; no log with image-derived data; backend/app changed only for WP4a) is met for what I inspected; `backend/app/` changes were not diffed (no VCS), only the two named files were read.

## 6. Missing protections
- Conversion of decompress/unpack errors to FetchError and cancellation of queued work (F1).
- A wall-clock cap per request (F3).
- No integrity pin beyond Content-Length and ETag notes: ETags are recorded in the source but not checked, and the content hash of the archives is not pinned. A hostile server could serve altered content of the same length. For a read-only evaluation of aggregate numbers this is acceptable.

## 7. Deferred risks
Hash-pinning dataset members; scanning outputs for personal data before committing; legal review of biometric/dataset terms (already a listed blocker); the pre-existing production items (F1 concurrency memory, F2 body cap, rate limiting, authentication) are outside this scope and are unchanged.

## 8. Overall assessment
PASS. No critical or high finding and no blocker. Two low findings (F1 robustness, F2 home path in a doc) should be fixed in a follow-up; neither blocks.

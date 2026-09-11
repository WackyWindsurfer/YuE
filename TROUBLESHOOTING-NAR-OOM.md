# YuE2 Long-Song CUDA OOM — Fix Instructions

**Symptom:** `yue2.cli generate` completes the AR phase, then dies in the NAR
"Synthesizing audio" stage on a ~32 GiB GPU (verified: RTX 5090):

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.33 GiB.
GPU 0 has a total capacity of 31.84 GiB of which 2.43 GiB is free.
... Of the allocated memory 27.22 GiB is allocated by PyTorch ...
```

The stack trace ends in `src/yue2/nar.py` → `CachedNAR` → `attention`
(`F.scaled_dot_product_attention`). Short tracks (<~3 min) render fine;
longer tracks OOM.

> **Root cause in one line:** the NAR attention call processes *all* query rows
> in one SDPA call, materializing a ~2.3 GiB temporary block that doesn't fit
> alongside the resident AR model + activations. It's a VRAM-headroom problem,
> not a bug.

---

## 1. Confirm it's the NAR stage

Check the traceback:

- Ends in `nar.py` → `attention` / `F.scaled_dot_product_attention`
  → **this is the problem described here.**
- Ends in `sampling.py` → `generate_tokens` (the AR phase)
  → different issue; these instructions won't help.

## 2. Skip the levers that do NOT work (verified 2026-09-11)

| Lever | Result |
|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **no-op on Windows** (PyTorch emits a warning and ignores it) |
| `max_split_size_mb:512` | still OOM |
| `--offload-ar` | reduces AR memory (27.2→20.7 GiB) but still OOM |
| higher `--budget` | still OOM |
| `--quantization fp8` | only affects the AR phase, not the NAR stage |

Do not spend time on these. Go straight to the fix.

## 3. Apply the one-block patch to `src/yue2/nar.py`

The upstream code **already** has the `query_chunk_size` parameter, the
`block = query_chunk_size or (...)` line, and the
`for start in range(0, len(q), block)` tiling loop. The only missing piece is
reading the environment variable when the argument is `None`.

In `attention()`, change:

```python
    if query_chunk_size is None:
        pass
```

to:

```python
    if query_chunk_size is None:
        # Local patch: allow bounding NAR attention temp storage for long-song
        # renders on 32 GiB cards via env var. Inert unless set.
        import os
        try:
            query_chunk_size = int(os.environ.get("YUE2_QUERY_CHUNK_SIZE", "0")) or None
        except ValueError:
            query_chunk_size = None
```

That is the entire patch. It is **inert unless the env var is set**, so normal
short-song runs are byte-for-byte unaffected.

**Why it's safe:** the code docstring states query tiling "changes temporary
storage, never the visible key set" — tiling only changes how the computation
is chunked, not what it computes. Output is numerically identical; only peak
memory changes (the temp block drops from ~2.33 GiB to ~0.3 GiB at 1024).

## 4. Render long tracks with the env var set

```bash
YUE2_QUERY_CHUNK_SIZE=1024 .venv\Scripts\python.exe -m yue2.cli generate \
  --request <req.json> --abc-file <score.abc> --cot melody \
  --output <fresh dir> --backend torch --budget 31
```

- **1024** is the value that worked on a 32 GiB card.
- Less VRAM? Try **512**.
- More VRAM and still OOM? The constraint is the other resident tensors, not
  the block size — this patch won't help.
- Tracks under ~3 min don't need the env var at all.

**Verified:** a 333 s track completed (32/32 NAR steps, 9/9 VAE chunks) at
~18 GiB peak, no truncation, status complete.

## 5. Use a fresh output directory per render

A stale `failure.json` in the target directory fails the run immediately —
unrelated to OOM, but a common confounder when retrying after a crash.

---

## Caveats

- **6+ min tracks** can additionally hit the 9000-token semantic limit
  (`truncated.semantic=true` in `result.json`). That is a separate cap, not
  OOM. If the ending sounds abrupt, re-render with a higher cap.
- **The patch is local, not upstreamed** — re-apply it after any repo update
  or reinstall.
- **`expandable_segments` is a Windows-specific no-op.** On Linux the CUDA
  allocator can defragment, so the OOM may not occur there at all — the patch
  is still harmless there, but may be unnecessary.
- **Unrelated but adjacent:** `--backend torch` (CUDA graphs) used to crash on
  official Windows torch 2.10.0 wheels with `RuntimeError: USE_FLASH_ATTENTION
  was not enabled for build` (the flash-attention op is registered but a stub
  that raises, so the old `hasattr` check was a false positive). Fixed per
  upstream PR #166: a `_flash_attention_available()` probe using
  `torch.backends.cuda.can_use_flash_attention` in `src/yue2/cuda_graph.py`.
  If you see that crash, apply that fix or use `--backend torch-eager`.

## Reproduction / verification

```bash
# 1. Apply the patch (section 3)
# 2. Run a long track (>3 min) with the env var:
YUE2_QUERY_CHUNK_SIZE=1024 .venv\Scripts\python.exe -m yue2.cli generate \
  --request <req.json> --abc-file <score.abc> --cot melody \
  --output <fresh dir> --backend torch --budget 31
# 3. Success = final log line:
#    {"status": "complete", ..., "truncated": {"abc": false, "semantic": false}}
#    and an audio.flac in the output dir.
```

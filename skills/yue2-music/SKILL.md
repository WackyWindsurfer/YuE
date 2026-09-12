---
name: yue2-music
description: Generate, cover, transcribe, and edit songs with YuE2 and SheetSage2/MERT2. Use for YuE2 full/melody/off generation, audio-to-ABC covers, style or lyric changes, score editing, agentic reharmonization, melody preservation, singable lyric adaptation, and reproducible listening comparisons; also for YuE2 生成、翻唱、改编、改谱、换词和智能体编辑.
---

# YuE2 Music

Turn a musical request into a reproducible song and an audible comparison. Use released
model interfaces. Retain an original song and its plan before making changes.

## Choose the workflow

| Request | Workflow |
| --- | --- |
| Generate with editable melody and harmony | YuE2 `cot="full"` → ABC → song |
| Generate with a melody plan and free accompaniment | YuE2 `cot="melody"` → chord-free ABC → song |
| Generate without symbolic planning | YuE2 `cot="off"` → song; no editable ABC |
| Cover a recording | SheetSage2 → inspect/correct ABC → strip chords → YuE2 `melody` |
| Cover an ABC melody | Inspect/convert native ABC → strip chords → YuE2 `melody` |
| Change harmony, instruments, tempo, structure, or lyrics | Copy full plan → edit ABC/text → regenerate |
| Agentic editing | Export plan/baseline → bounded editing agent → check invariants → render → compare |
| Analyze musical features | Use MERT2 only when continuous features are needed |
| Generate with the GGUF / low-VRAM engine (Q8/Q4) | audio.cpp GGUF CLI — read [gguf-engine.md](references/gguf-engine.md) |

```text
audio → SheetSage2 [loads MERT-v2-FullSong itself] → ABC
style + lyrics → YuE2 full/melody planning        → ABC
                                                  edit/validate
style + lyrics + ABC → YuE2 semantic generation → synthesis → latents → VAE → song
style + lyrics      → YuE2 off generation      → synthesis → latents → VAE → song
```

Do not feed public MERT feature tensors to YuE2 as codec tokens. YuE2 exposes no
audio-reference, phoneme-alignment, or local-inpainting argument.

## Choose the engine

Two engines are available on this host; pick per request (the user may name one explicitly):

- **torch** (default): `yue2` CLI or `run_yue2.py` helper. **The default CUDA-graph backend now works on this host** (the `cuda_graph.py` flash-attention fallback fix is committed on branch `agent/yue2-flash-fallback-and-nar-oom-fix` and active in the editable install) — it probes `is_flash_attention_available()` and falls back to cuDNN/SDPA instead of crashing. Verified: 63.1 s of audio in 31.3 s on the default backend. Use for editable plans, ABC export, plan-only, and the normal workflow. `--backend torch-eager` remains a safe fallback if the graph path ever misbehaves.
- **audio.cpp GGUF**: standalone CLI (`D:\AI\audio.cpp\build\windows-cuda-release\bin\audiocpp_cli.exe`) with the GGUF bundle in `D:\AI\YuE\models\Yue2-3B-GGUF`. Use when the user asks for GGUF/Q8/Q4/low-VRAM. **Not available for plan-only or ABC export.** See [gguf-engine.md](references/gguf-engine.md) for the exact command and request-JSON shape.

**Model paths (torch engine):** the local `D:\AI\YuE\models\Yue2-3B` and `models\YuE2-Vae` dirs are **incomplete** (weights only, no `qwen.tiktoken`/`config.json`), so the CLI's local-dir auto-detection fails. Either let the CLI fall back to the HF cache (`m-a-p/YuE2-3B` / `m-a-p/YuE2-Vae` — the default when the local dir is absent), or pass the full HF snapshot dirs:

```bat
--model C:\Users\michi\.cache\huggingface\hub\models--m-a-p--YuE2-3B\snapshots\1a96eca688d6ae5d7f0feb88573fec89920fcd19
--vae C:\Users\michi\.cache\huggingface\hub\models--m-a-p--YuE2-Vae\snapshots\95535e72a97bc0f09b8ada125d26b4009428c0e8
```

Both engines write to `D:\AI\output\YuE\<name>\`. Torch results carry structured truncation flags; GGUF results do not — mark them "needs review" and listen for a complete ending.

## Set up the needed models

Read [models-and-setup.md](references/models-and-setup.md). Install the YuE2 runtime
from the official GitHub repository source. Use a separate environment
for SheetSage2 because dependency pins differ. Download the public model snapshots and
record their revisions. This skill's original instructions, helpers, and templates are
licensed under [Apache 2.0](LICENSE). Copyright (c) 2026 the YuE2 authors.
Model weights and third-party dependencies retain their applicable licenses.

Use the supported baseline: one request at a time, BF16-capable NVIDIA GPU with 24 GB
VRAM, default YuE2 settings. Do not silently shorten a requested song or lower inference
settings to hide an OOM. Free allocations or choose suitable hardware; report changes.

Use `YuE2-Vae` for listening and `YuE2-Vae-legacy` when reproducing the supplied benchmark
protocol. Keep decoded files separate. Do not infer their roles from the word “legacy.”

Run helper paths below relative to this skill folder, with the appropriate environment's
Python. Select snapshots using `--model`, `--revision`, and `--vae-revision`, or local
model directories plus `--offline`. Use fresh output directories.

## Generate and retain the plan

Start with [assets/prompt.json](assets/prompt.json), an original example. Put genre,
instruments, vocal character, language and intended tempo in `style`; put section tags
and actual words in `lyrics`. Keep implementation notes out of lyrics.

```bash
python scripts/run_yue2.py generate --request assets/prompt.json --output outputs/pop
python scripts/run_yue2.py all-modes --request assets/prompt.json --output outputs/modes
python scripts/run_yue2.py plan --request assets/prompt.json --output outputs/plan
```

Inspect `result.json`, truncation, `score.abc`, `request.json`, and audio. Keep exact
tokens and `latent.npy`; the helper saves native artifacts. Preserve all requested modes
and failures. A successful process or playable file does not establish musical quality.

Read [generation-and-covers.md](references/generation-and-covers.md) for Python, CLI,
exact plan continuation, CFG, sampling, and cached decoding. Load unchanged plans with
`SymbolicPlan.load`; submit modified ABC as a new input. CLI `--resume` verifies a
completed result; it does not continue interrupted generation.

## Cover a recording

1. Transcribe in the SheetSage2 environment. Select the vocal melody or the full lead
   melody, including instrumental passages.
2. Inspect warnings and correct missed notes, meter or key before attributing errors
   to YuE2. Preserve source audio and raw transcription.
3. Export chord-free ABC. Select a retained voice explicitly when dropping a part;
   removing chords alone should preserve both melodic voices and their rests.
4. Render with `cot="melody"`, target style and suitable lyrics. This supplies a symbolic
   melody condition; it does not preserve the source singer's identity or waveform.

```bash
# SheetSage2 environment.
python scripts/transcribe.py reference.wav --task melody-full --output outputs/transcription
python scripts/abc_tools.py strip-chords outputs/transcription/score.abc outputs/cover.abc

# YuE2 environment; the request supplies target style and lyrics.
python scripts/run_yue2.py generate --request assets/prompt.json --cot melody \
  --abc-file outputs/cover.abc --output outputs/cover-song
```

`cot="melody"` does not remove chord symbols automatically. To retain the original
harmony as well, use full transcription and `cot="full"`; call this score-conditioned
regeneration with melody and harmony.

## Edit or delegate an edit

Read [editing-workflows.md](references/editing-workflows.md) and
[abc-editing.md](references/abc-editing.md) before changing a score.

1. Render a baseline from the full plan. Freeze its original directory.
2. Define invariants: exact pitches; pitch plus rhythm; contour only; or bounded melodic
   adaptation. Specify voices/passages, lyrics, instruments, tempo, meter and structure.
3. If delegation is available, give a score-editing agent raw ABC, prompt, lyrics, the
   requested change and the [edit brief](assets/edit-brief.md). Request a new ABC,
   revised style/lyrics as needed, and an edit manifest. Give a separate reviewer the
   before/after artifacts and constraints. Without delegation, perform these stages
   yourself. Keep model generation sequential per GPU.
4. Check musical events, not character strings: ties, accidentals and compressed rests
   matter. Run:

   ```bash
   python scripts/abc_tools.py inspect edits/jazz.abc
   python scripts/abc_tools.py compare outputs/plan/score.abc edits/jazz.abc --voices Vocal
   python scripts/run_yue2.py generate --request edits/jazz.json --cot full \
     --abc-file edits/jazz.abc --output outputs/jazz
   ```

   Add `--allow-tempo-change` for intentional tempo changes. Exact comparison should
   fail for intentional rhythm changes; audit permitted differences from its report
   instead of relabeling the result “melody preserved.”
   Keep the edited score connected through `--abc-file` or request `abc_path`;
   omitting both with `abc: null` generates a fresh plan and discards the edit.
5. Regenerate after changing style, lyrics or ABC. Old acoustic latents can be decoded
   again, but cannot implement a musical or lyric edit.
6. Compare full songs and short passages around the edit. Revise when the requested
   effect fails; retain each attempt and its actual prompt.

For lyric translation, adapt syllables, stress, vowels and breath points. Keep a
syllable/phoneme-to-note sidecar. Do not invent a `phonemes` field or mistake the sidecar
for hard acoustic alignment. Use ASR/PER and listening as separate evidence.

## CUDA OOM on long songs (this host: RTX 5090, 32 GiB)

Symptom: `generate` completes the AR phase (8320 tokens) then dies at
"Synthesizing audio" with `OutOfMemoryError` (e.g. "Tried to allocate 2.33 GiB
... 27.22 GiB is allocated by PyTorch"). Cause: the NAR acoustic stage holds
the AR model + a KV prefix cache that grows with song length; a long track
(333 s) pushes base allocation to ~27 GiB, and the final attention step needs a
2.33 GiB contiguous block. On this Windows platform `expandable_segments`
(the standard anti-fragmentation fix) is a **no-op**, so the allocator cannot
defragment to place that block even though total free > needed. Short tracks
(171 s) stay under the ceiling and fit. **Not** caused by other GPU processes:
ComfyUI idling on this host holds no VRAM (nvidia-smi showed 0 MiB attributed)
— do not kill it to "free headroom"; it was never the constraint.

What does NOT work here (verified 2026-09-11):
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — **no-op on Windows**
  (warning: "expandable_segments not supported on this platform").
- `max_split_size_mb`, `--offload-ar`, raising `--budget` to 31 — all still OOM.

Working fix: query tiling. The library's `query_chunk_size` bounds the NAR
attention temp storage without changing results ("changes temporary storage,
never the visible key set"). The CLI does not expose it, so a local patch in
`D:\AI\YuE\src\yue2\nar.py` (in `attention()`) reads
`YUE2_QUERY_CHUNK_SIZE` from the environment when the arg is None (inert
unless set). Run long-song renders with:

```bash
YUE2_QUERY_CHUNK_SIZE=1024 .venv\Scripts\python.exe -m yue2.cli generate \
  --request <req.json> --abc-file <score.abc> --cot melody \
  --output <fresh dir> --backend torch-eager --budget 31
```

Verified: 333 s grunge cover completed (32/32 NAR steps, 9/9 VAE chunks) at
~18 GiB used. If the YuE repo is updated/reinstalled, re-apply the patch or
pass `query_chunk_size` programmatically.

## Deliver an audible result

Read [listening-and-evaluation.md](references/listening-and-evaluation.md). Return playable
audio, full prompt/lyrics, before/after ABC, invariant checks and requested evaluations.
Keep model/decoder identity and failures visible.

```bash
python scripts/listen.py outputs/pop outputs/jazz --output outputs/comparison
```

This creates a local HTML player, copies audio, and includes the exact requests. It does
not publish or upload. Distinguish symbolic checks, ASR, listening and quality scores.
Deliver custom edit manifests and before/after comparison reports alongside the page;
the player copies a fixed set of native artifacts, not arbitrary sidecars.
Do not claim exact note realization, instrument removal or sample-accurate preservation
from an ABC check or SongBench score alone.

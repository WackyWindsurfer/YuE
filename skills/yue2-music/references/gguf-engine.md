# audio.cpp GGUF engine (local setup)

A second, standalone engine for YuE2 generation: the **audio.cpp** C++ CLI running the GGUF model bundle. It is fully independent of the PyTorch (torch) engine and of the Studio UI — one fresh process per song.

## When to use it

- User asks for a **GGUF / Q8 / Q4 / low-VRAM** generation, or wants the smaller/faster engine
- Torch engine unavailable or OOM (GGUF Q8+F16 peaks ~8.9 GB, Q4+F16 ~7.8 GB — publisher numbers, not guarantees)
- **Not available for:** Plan-only, generated-ABC export, or any workflow needing the editable score → use torch for those. Supplied-ABC covers work on GGUF; source transcription still uses SheetSage2.

## Local paths (verified on this host)

| Component | Path |
|---|---|
| CLI executable | `D:\AI\audio.cpp\build\windows-cuda-release\bin\audiocpp_cli.exe` |
| GGUF model dir | `D:\AI\YuE\models\Yue2-3B-GGUF` |
| Main models | `yue2-3b-q8_0.gguf` (balanced), `yue2-3b-q4_0.gguf` (if present), `yue2-3b-bf16.gguf` |
| VAE | `yue2-vae-f16.gguf` |
| Sidecars (all four required) | `sidecars/yue2-model-config.json`, `sidecars/yue2-generation-config.json`, `sidecars/yue2-qwen.tiktoken`, `sidecars/yue2-vae-config.json` |
| Output convention | `D:\AI\output\YuE\<name>\` (same as torch) |

## Run one song

1. Write a request JSON file (UTF-8; **all option values are strings** — audio.cpp parses JSON numbers as doubles, so 63-bit seeds must be strings):

```json
[{"id": "song", "text": "[Verse]\n…lyrics…\n\n[Outro]\n…",
  "options": {"style": "…", "cot": "full", "seed": "42", "num_inference_steps": "32"}}]
```

2. Run (one fresh process per song; keep model files untouched while running):

```bat
D:\AI\audio.cpp\build\windows-cuda-release\bin\audiocpp_cli.exe --task gen --family yue2 ^
  --model D:\AI\YuE\models\Yue2-3B-GGUF --backend cuda --device 0 --threads 4 ^
  --request-sequence <request.json> --batch-merge-audio concat ^
  --out D:\AI\output\YuE\<name>\audio.wav --log --metrics
```

3. Check `--metrics` output (wall_ms, audio_duration_ms, x_realtime) and verify the WAV (duration, non-silent). The original WAV is the deliverable; the Studio normally converts it to FLAC for its player — for direct CLI use, the WAV is fine.

## Verified reference run

Short `full`-planning song, Q8_0 + F16 + CUDA + 32 ODE steps: **52.7 s of 48 kHz stereo audio in 11.2 s (4.7× realtime)**, peak 0.891.

## Limits & reporting

- Output has **no structured truncation flags** (the CLI does not return them) — listen for a complete ending, especially if max tokens changed. Mark results "needs review" when reporting.
- `cot` values: `full` / `melody` / `off` are forwarded; a supplied `abc` (score-conditioned) is supported via the request (the Studio writes it to a `score.abc` file and passes `abc_file`).
- PyTorch settings (model/VAE paths, device budget, quantization, offload, revisions, cache) are **not used** by this engine.
- Q8 is the balanced default; Q4 below ~12 GiB VRAM. Q4/BF16 generation is less tested.
- Engine provenance: audio.cpp dev commit `fbe3eed` (the pinned contract). If the build is missing or broken, the rebuild steps are in the Obsidian note "Yue2 Studio + audio.cpp GGUF Setup".

## Engine choice summary

| Need | Engine |
|---|---|
| Normal song, editable plan, ABC export, plan-only | **torch** (`--backend torch-eager`) |
| GGUF / low-VRAM / Q8 or Q4 | **audio.cpp GGUF** (this reference) |
| Score-conditioned cover (supplied ABC) | either |
| Source transcription | SheetSage2 (separate env) — then either engine |

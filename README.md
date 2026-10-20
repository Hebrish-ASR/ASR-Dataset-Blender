---
title: ASR Dataset Blender
emoji: 🎚️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.29.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
hf_oauth: true
hf_oauth_scopes:
  - read-repos
short_description: Blend speech datasets into NeMo training manifests
---

# ASR Dataset Blender

Pick speech datasets from a library, filter them, hold out validation and test data, set how
much each source contributes, and download the blend: NeMo manifests, a NeMo Lhotse
`input_cfg.yaml`, a reproducible recipe, and a script that fetches the audio.

- **Space (demo):** <https://huggingface.co/spaces/Hebrish/ASR-Dataset-Blender>
- **Library:** the datasets in [`library.json`](library.json), all recorded by Daniel Rosehill for the
  [English-Hebrew code-switching Parakeet fine-tune](https://github.com/Hebrish-ASR/English-Hebrew-Codeswitching-Parakeet-Finetune)
  ([collection](https://huggingface.co/collections/Hebrish/parakeet-v2-fine-tune-training-data-6ad63c739d300685a62edb3e)).
  Any Hugging Face dataset with the same layout can be added.

## Why this exists

As of 2026-10-20 there is no standard tool for this. The pieces that exist are mixing
*mechanisms*, not blend *builders*:

- NeMo's Lhotse dataloader takes an `input_cfg` list of sources with a `weight` each, with
  temperature reweighting ([docs](https://docs.nvidia.com/nemo/speech/latest/dataloaders.html)).
  It samples at training time and assumes the per-source manifests and held-out sets already exist.
- Hugging Face `datasets` has `interleave_datasets(probabilities=...)` and `concatenate_datasets`,
  with no held-out or term-aware splitting.
- Megatron-style "data blends" (`--data-path 0.3 a 0.7 b`) are for text pretraining.

The Blender handles the step before those: choosing rows, de-duplicating sources that overlap,
holding data out (including term-disjoint hold-out for vocabulary work), sizing each source, and
recording all of that with every source pinned to a commit so the blend can be rebuilt exactly.
Its output feeds NeMo directly, either as one shuffled `train.jsonl` or as `input_cfg.yaml`.

## Cost and access

The Space runs on free CPU (`cpu-basic`). It reads only each source's `metadata.jsonl` (a few MB);
the audio is downloaded by `fetch_audio.py` on the user's own machine. It calls no inference API,
so nobody's blend costs the Space owner anything.

Gated sources are read with the **visitor's own** token from "Sign in with Hugging Face"
(`hf_oauth`, scope `read-repos`). The Space holds no token of its own, so the gate still decides
who can blend which rows. A visitor without access sees "no access" for that source.

## Source layout

Every source must use the Hugging Face audiofolder layout:

```
data/<split>/metadata.jsonl   # one row per clip: file_name, text, duration_s (required); id,
                              # categories, speaker, target_terms (used when present)
data/<split>/<file_name>      # the audio
```

`id` defaults to the file name's stem. A row with no `speaker` is taken to be the library's
`default_speaker`.

## How a blend is built (`blender/core.py`)

1. **Resolve.** Each source's revision (default `main`) is pinned to its commit sha.
2. **Filter.** Categories (include any / exclude any), speakers, clip duration.
3. **De-duplicate** (on by default). An `id` seen in an earlier source is dropped. Order matters:
   list the source whose copy you want first.
4. **Hold out**, before any sizing, so caps and shares never change val or test:
   - `mode: "source"` (default): each source's own non-train splits become test; `val_pct` of
     train becomes validation.
   - `mode: "resplit"`: everything is pooled and split anew by `test_pct` and `val_pct`.
   - `group_by: "id"` assigns each row by a hash of its id and the seed.
     `group_by: "target_terms"` holds out *terms*: a held-out term sends every row containing it
     to val/test, so no held-out term is ever trained on. That is the rule the Hebrish sentence
     bank uses. A row's other terms may still appear in train. Rows without terms fall back to id.
5. **Size train per source.** Each source takes all its rows, or `cap_min` minutes, or a `share`:
   - Some sources with shares and some without: the unshared sources fix the minutes, and the
     total is `fixed / (1 - sum of shares)`.
   - Every source with a share: shares are relative weights, and the total is the largest that
     no source has to repeat rows for. With `oversample`, the largest source sets the total
     and smaller sources repeat rows.
   - A share its source cannot fill is reported as a warning. Rows are taken in seeded random
     order. Oversampling more than 3x also warns.
6. **Bundle.** A zip containing:

| File | What |
|---|---|
| `manifests/{train,val,test}.jsonl` | NeMo manifests: `audio_filepath` (relative), `duration`, `text`, `id`, `source`, `audio_url` (pinned), plus the source's own fields |
| `manifests/train_by_source/*.jsonl` | train split per source, for `input_cfg.yaml` |
| `input_cfg.yaml` | NeMo Lhotse sources, weighted by each source's share of train minutes |
| `blend.json` | the full recipe (sources pinned to shas) and the report: splits, per-source mix, warnings |
| `fetch_audio.py` | downloads exactly the referenced files at the pinned shas (`HF_TOKEN` for gated sources) |

A repeated row (oversampling) carries `"repeat": n`.

## Recipe

```json
{
 "name": "cs-like-r6",
 "sources": [
  {"repo": "danielrosehill/English-Hebrew-Codeswitching-Sentences"},
  {"repo": "danielrosehill/General-English-Sentences", "share": 0.2, "cap_min": null, "revision": "main"}
 ],
 "filters": {"include_categories": [], "exclude_categories": ["singing"], "speakers": ["Daniel Rosehill"],
             "default_speaker": "Daniel Rosehill", "min_duration_s": 0, "max_duration_s": 0},
 "dedupe": true,
 "held_out": {"mode": "source", "val_pct": 6.25, "test_pct": 10, "group_by": "id"},
 "oversample": false,
 "text_field": "text",
 "seed": 7
}
```

Defaults are in `DEFAULT_RECIPE` in `blender/core.py`. The `blend.json` from any bundle is itself
a valid recipe.

## CLI

```bash
uv venv && uv pip install huggingface_hub
python -m blender --scan                                  # sizes, categories, speakers per source
python -m blender --recipe recipes/cs-like-r6.json        # writes cs-like-r6.zip
python -m blender --recipe old-bundle/blend.json          # rebuild a past blend exactly
```

It uses `HF_TOKEN` or the cached `hf auth login`.

## Space

`app.py` is the Gradio front end for the same `blender.core`. Run it locally with
`uv pip install "gradio[oauth]==6.29.0" huggingface_hub && python app.py`. Locally, sign-in is
mocked and the cached token is used. Deploy by uploading the repo (minus `.venv`, `tests`) to the
Space `Hebrish/ASR-Dataset-Blender`. `scripts/deploy_space.py` does this.

## Adding a dataset to the library

Add `{"repo", "title", "note", "gated"}` to `library.json` and redeploy the Space. The library is
only the list of offered sources. A recipe can name any dataset the token can read.

## Tests

`uv pip install pytest && python -m pytest -q`. The tests are offline and use synthetic sources.

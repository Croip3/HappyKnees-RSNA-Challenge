# RSNA Knee Abnormality Detection

Predicts 12 findings per knee MRI study (ACL, MCL, both menisci, three OA compartments, effusion,
synovitis, Baker's cyst, contusion, fracture). Scored by macro ROC-AUC.

## Idea

- **Labels from reports.** Only 58 training studies have gold labels, but every study has a radiology
  report. An LLM (Qwen3.8-27B) marks each finding in the report as positive, negative, uncertain or not
  mentioned. These states become soft targets with per-state weights. The 58 gold studies are used to
  tune the label definitions and to validate.
- **Frozen backbone, small head.** A frozen DINOv2 embeds every slice once, and only a small attention
  model is trained (5 folds in about 1 minute on one GPU). The ~570 GB of DICOMs are processed in a
  single pass on Kaggle.

## Architecture

```
DICOM ──► preprocess (uint8 stacks per plane) ──► frozen backbone (per-slice CLS + mean-patch) ──┐
reports ─► LLM (states per finding) ─► gold validation ─► soft targets + weights ────────────────┼─► cross-slice attention head (5 folds)
                                                                                                 └─► offline Kaggle notebook ─► submission.csv
```

1. **Preprocess:** one series per plane (sagittal, coronal, axial), preferring fluid-sensitive
   fat-suppressed sequences, resampled to 448 px uint8 slice stacks.
2. **Backbone:** DINOv2 ViT-S/14 with registers, frozen. Each slice gives its CLS token plus the mean
   patch token (optionally a 4x4 grid of pooled patch tokens).
3. **Cross-slice model** ([cross_slice.py](src/knee/models/cross_slice.py), d = 256):
   - slice token = projected features + slice-position, plane and sequence-type embeddings
   - 2 within-plane transformer layers; each slice attends to its ±2 neighbours
   - 1 fusion layer across all planes
   - per-finding attention pooling: one learned query per finding gives 12 logits
4. **Inference:** 5-fold ensemble in an internet-off Kaggle notebook.

## Results

Run `full_v1`: out-of-fold predictions from 5 folds on about 1,870 LLM-labelled studies, with a
224 px backbone input.

| Label | AUC vs LLM | AUC vs gold |
|---|---|---|
| ACL | 0.753 | 0.760 |
| MCL | 0.706 | 0.744 |
| Medial Meniscus | 0.755 | 0.639 |
| Lateral Meniscus | 0.710 | 0.677 |
| Medial OA | 0.836 | 0.851 |
| Lateral OA | 0.824 | 0.810 |
| PF OA | 0.784 | 0.740 |
| Effusion | 0.912 | 0.824 |
| Synovitis | 0.780 | 0.771 |
| Baker's | 0.713 | 0.781 |
| Contusion | 0.769 | 0.702 |
| Fracture | 0.796 | 0.583 |
| **Macro** | **0.778** | **0.740** |

The gold column covers only 58 studies, so per-label gold AUCs are noisy. Feeding the backbone at
448 px did not help (0.773 vs LLM, 0.708 vs gold).

## Configuration

Everything is driven by `configs/default.yaml`. Any value can be overridden on the command line
with `--set section.key=value` (repeatable) or a YAML file passed with `--config`. Every stage
skips outputs that already exist (`--overwrite` recomputes them) and accepts `--limit N`.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[decode,labels,dev]"
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data/raw   # + the other CSVs
```

The DICOMs (~570 GB) are only used on Kaggle (`/kaggle/input/competitions/rsna-knee-abnormality-detection`);
`paths.data_root: auto` finds them there and falls back to `data/raw` locally.

## Stages

| Stage | Command | Output |
|---|---|---|
| 0. Inspect data | `python scripts/inspect_data.py [--header-scan -1]` | `results/inspect/` |
| 1. Preprocess DICOM → uint8 cache | `python scripts/preprocess.py --split train` | `data/cache/<split>/<study>.npz` + `manifest.csv` |
| 2a. LLM report labels | `python scripts/llm_extract.py [--limit 50]` | `data/labels/llm_labels.jsonl` |
| 2b. Gold agreement | `python scripts/validate_labels.py` | `results/labels/gold_agreement.csv`, `disagreements.csv` |
| 2c. Final targets | `python scripts/build_targets.py` | `data/labels/labels_final.csv` |
| 3. Feature cache | `python scripts/extract_features.py --split train` | `data/features/<tag>/<split>/<study>.npz` |
| 1+3 streaming (Kaggle) | `python scripts/prep_embed.py --split train [--save-cache]` | features (+ cache) |
| 5. Train | `python scripts/train.py` | `runs/<run>/` (log, `metrics.csv`, `fold*/best.pt`, `oof.csv`) |
| 6. Predict | `python scripts/predict.py --set infer.checkpoints=runs/<run>` | `submission.csv` |
| Kaggle bundle | `python scripts/make_kaggle_bundle.py --run runs/<run> [--push-bundle --push-kernel]` | datasets + `notebooks/kaggle_inference.ipynb` |
| Any stage on Kaggle | `python scripts/kaggle_kernel.py --name X --gpu --internet --cmd "scripts/..." --push` | private script kernel |

Switching the model only takes config changes, for example:

```bash
python scripts/train.py --set model.slice_context=all          # full attention instead of +-2 slices
python scripts/train.py --set model.slice_context=3 --set model.head=mean
python scripts/prep_embed.py --set backbone.name=dinov2_vitb14_reg   # then train with the same --set
```

## Training notebook

`notebooks/kaggle_training.ipynb` runs the training side in one place, on Kaggle or locally in VS Code:
inputs -> features (reuses the attached `knee-full-prep` output, otherwise computes them) -> LLM vs gold
agreement and label statistics -> K-fold training with the verbose logs -> out-of-fold AUCs, learning
curves and checkpoints. Edit the parameters cell at the top (run name, config, overrides, folds, epochs).

```bash
# publish code + current labels as private datasets and run it on Kaggle (T4, internet off)
python scripts/push_training_notebook.py --push-src --push-labels --push-kernel
# quick check without editing the notebook: parameters are injected after the parameters cell
python scripts/push_training_notebook.py --push-kernel --param "RUN_FOLDS = [0]" --param "EPOCHS = 2"
```

Its `runs/<RUN_NAME>` output feeds `scripts/make_kaggle_bundle.py --run ...` for the submission notebook.

## Running on Kaggle (the DICOMs only exist there)

`scripts/kaggle_kernel.py` embeds `src/`, `scripts/` and `configs/` into a private script kernel, so a
code change needs no dataset upload. Outputs go to `/kaggle/working`.

```bash
# full train set: one 448 px preprocessing pass, three feature variants (~70 min on 2x T4)
COMMON="--config configs/full.yaml --set paths.cache_dir=/kaggle/working/cache --set paths.features_dir=/kaggle/working/features --set preprocess.num_workers=4 --variant backbone.image_size=448 --variant features.patch_grid=4"
python scripts/kaggle_kernel.py --name knee-full-prep --gpu --internet --cmd "scripts/prep_embed.py --split train $COMMON" --push
# pack the compact feature folders into ~100 MB tars (the CLI drops large downloads)
python scripts/kaggle_kernel.py --name knee-pack-features --no-competition --kernel-source <user>/knee-full-prep \
    --cmd "scripts/pack_outputs.py --pattern 'features/*_g0_*' --out /kaggle/working/packed" --push
kaggle kernels output <user>/knee-pack-features -p data/kaggle_full    # then: tar -xf <part>.tar -C data
```

Inference: `scripts/make_kaggle_bundle.py` uploads `knee-weights` (backbone weights) and
`knee-inference` (source + `fold*/best.pt`) as private datasets and pushes the internet-off notebook
`knee-infer` (`notebooks/kaggle_inference.ipynb`). It prints a runtime breakdown and writes
`/kaggle/working/submission.csv`; `KNEE_DRY_RUN=5` limits it to the first 5 test studies.

## LLM labels

The LLM runs at training time only (the test set has no reports). Qwen3.8-27B (`Qwen/Qwen3.8-27B`,
Apache 2.0) is a 27.8B vision-language model with a hybrid Gated-DeltaNet/attention stack. Its 4-bit
AWQ/GPTQ builds for vLLM are 19.5-22 GB (bf16 embeddings, LM head and vision tower), so they do not fit
a 16 GB GPU. What works:

| Where | Backend | Model | Speed |
|---|---|---|---|
| local 16 GB GPU | `ollama` | `qwen3.8:27b` (Q4_K_M, MTP drafting, 33% on CPU) | ~3.4 reports/min |
| Kaggle 2x T4 | `ollama` via `scripts/ollama_setup.py` | `qwen3.8:27b` fully in VRAM | see kernel log |
| any (optional) | `openai` via `scripts/serve_llm.sh` (llama.cpp Docker) | `unsloth/Qwen3.8-27B-GGUF` UD-IQ4_XS | untested here |

```bash
python scripts/llm_extract.py --limit 50                            # prints reports/min and the ETA for all
python scripts/llm_extract.py --studies-file configs/llm_bulk_all.txt --reverse --max-minutes 650   # 2nd worker
python scripts/merge_llm_jsonl.py data/labels/llm_labels.jsonl <kaggle>/labels/llm_labels.jsonl --out data/labels/llm_labels.jsonl
python scripts/validate_labels.py && python scripts/build_targets.py
```

Output is JSON-schema constrained, temperature 0, thinking disabled, one retry with a repair prompt,
then `llm_error`. Results are appended per report, so every run resumes. Label definitions and the
post-operative rule live in `configs/label_definitions.yaml`; v2 adds severity thresholds learned from
the 58 gold studies (gold-LLM kappa 0.62 -> 0.66; v1 is kept as `label_definitions_v1.yaml`).
`targets.calibrate_all_states=true` replaces the fixed state targets by P(gold=1 | label, state).

## Tests

```bash
.venv/bin/python -m pytest -q
```

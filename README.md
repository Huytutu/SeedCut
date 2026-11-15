# SeedCut

A minimal implementation of **SeedCut**: a training-free re-look pipeline for visually grounded multiple-choice
VQA. The model looks at the regions its own uncertainty points at, then answers again from the full image and
one masked view.

## Layout

```
seedcut/uncertainty.py  answer-entropy gradient on the visual tokens (U)
seedcut/propose.py      Seed, Cut (half-maximum regions, ranking, nesting removal), phrase scores for Select
seedcut/semantic.py     LatentLens word bank, question phrases, patch z-scores
seedcut/pipeline.py     run_record: Select -> masked view -> answer, with full-image fallback
seedcut/model.py        Qwen2.5-VL loading, prompts, generation
seedcut/internvl.py     InternVL3-8B backend (up to 40 tiles)
seedcut/llava_next.py   LLaVA-1.6-Mistral-7B backend (AnyRes)
run.py                  one image, one question
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Word bank

LatentLens needs a contextual word bank from the backbone's own language model, at layer 24. For Qwen2.5-VL-7B,
download it from
[`McGill-NLP/contextual_embeddings-qwen2.5-vl-7b`](https://huggingface.co/McGill-NLP/contextual_embeddings-qwen2.5-vl-7b).
The repo holds layers 1, 2, 4, 8, 16, 24, 26 and 27 (28 GB); SeedCut only needs layer 24 (3.7 GB). Run this
from the root of this repository:

```bash
huggingface-cli download McGill-NLP/contextual_embeddings-qwen2.5-vl-7b \
    --include "layer_24/*" --local-dir data/latentlens/qwen2.5-vl-7b
```

With `huggingface_hub>=1.0`, use `hf download` with the same arguments.

This gives the layout SeedCut expects (`data/` is git-ignored):

```
SeedCut/
├── run.py
├── seedcut/
└── data/latentlens/qwen2.5-vl-7b/layer_24/embeddings_cache.pt
```

(sha256 of `embeddings_cache.pt`: `92cd6f0724dcda56838aadcf11de13157dee48246f73ed1e9633c7da4d0979e5`). On the first run SeedCut averages it
into word prototypes and saves `layer_24/prototypes.pt` next to it, so the folder must be writable. To keep the
bank elsewhere, set `SEEDCUT_BANK=<dir>` (the folder that contains `layer_24/`).

For InternVL3-8B and LLaVA-1.6-Mistral-7B, build the layer-24 bank with the LatentLens recipe (a text-only
forward of its concept corpus through the backbone's language model) and place it at
`data/latentlens/internvl3-8b/layer_24/` or `data/latentlens/llava-1.6-mistral-7b/layer_24/`.

## Run

Three V*Bench questions are in `examples/samples.json`; `--sample` downloads that one image from the
[`craigwu/vstar_bench`](https://huggingface.co/datasets/craigwu/vstar_bench) dataset and runs it:

```bash
python run.py --sample 0   # glove material (answer A); 1: dustpan color (C); 2: motorcycle left or right of the dog (A)
```

On your own image:

```bash
python run.py --image img.jpg --question "What is the color of the umbrella?" --options red blue green yellow
```

- `--model` takes `Qwen/Qwen2.5-VL-7B-Instruct` (default), `OpenGVLab/InternVL3-8B-hf` or
  `llava-hf/llava-v1.6-mistral-7b-hf`, as a hub id or a local path.
- `--prompt vstar|hrbench|mmerw` uses that benchmark's prompt and answer parser (default `vstar`).
- Qwen2.5-VL runs at its largest input, `--max-pixels 12845056` (12.8 MP). InternVL3 and LLaVA-1.6 run at
  their largest input by default.
- Decoding is greedy. At full resolution, the uncertainty backward uses gradient checkpointing and fits on two
  24 GB GPUs (`CUDA_VISIBLE_DEVICES=0,1`); LLaVA-1.6 fits on one.

The output lists the selected regions, the noun phrases, the masked view's box, the answer with the view
(`look_pred`) and the final answer (`final_pred`); if the view gives no parsable answer, the full-image answer
is used.

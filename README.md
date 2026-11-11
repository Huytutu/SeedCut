# SeedCut

A minimal core of **SeedCut**: a training-free re-look pipeline for visually grounded multiple-choice
VQA on Qwen2.5-VL-7B. The model looks at the regions its own uncertainty points at, then answers again.

## Method

1. **Relative uncertainty.** Entropy-gradient uncertainty backpropagates the answer entropy to the visual
   embeddings, for the question (`U`) and for a question-free prompt (`U0`). `U_rel = (U/ΣU) / (U0/ΣU0 + 1/|Ω|)`
   keeps the uncertainty this question adds.
2. **Seed.** Up to 20 local maxima of the 3×3-smoothed `U_rel`, at least 3 cells apart.
3. **Cut.** Around each seed, a normalized cut on the patch-affinity graph (layer-24 states, 8-neighbour,
   `exp(cos/τ)`) over three window sizes; the window whose region scores best is kept. Regions are ranked by
   `density · √mass · coherence`, with containment NMS, top five.
4. **Select.** LatentLens scores each region against every question noun, size-normalised against random
   translations of the same mask; a round-robin over the nouns picks two regions.
5. **Re-look.** One generation sees the full image, the enclosing crop and each region crop. An unparsed answer
   or a failed step falls back to the base answer.

## Layout

```
seedcut/propose.py      seeds, seeded normalized cut, coherence, region ranking, noun scores, round-robin
seedcut/pipeline.py     run_record: regions -> re-look -> fallback
seedcut/uncertainty.py  entropy-gradient uncertainty maps
seedcut/semantic.py     LatentLens noun targets and patch z-scores
seedcut/model.py        Qwen2.5-VL loading, prompts, generation
run.py                  one image, one question
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Word bank

LatentLens needs the contextual word bank of Qwen2.5-VL-7B's LLM at layer 24, from
[`McGill-NLP/contextual_embeddings-qwen2.5-vl-7b`](https://huggingface.co/McGill-NLP/contextual_embeddings-qwen2.5-vl-7b).
The repo holds layers 1, 2, 4, 8, 16, 24, 26 and 27 (28 GB); SeedCut only needs layer 24 (3.7 GB):

```bash
huggingface-cli download McGill-NLP/contextual_embeddings-qwen2.5-vl-7b \
    --include "layer_24/*" --local-dir data/latentlens/qwen2.5-vl-7b
```

With `huggingface_hub>=1.0`, use `hf download` with the same arguments.

This gives `data/latentlens/qwen2.5-vl-7b/layer_24/embeddings_cache.pt`
(sha256 `92cd6f0724dcda56838aadcf11de13157dee48246f73ed1e9633c7da4d0979e5`). On the first run SeedCut averages it
into word prototypes and saves `layer_24/prototypes.pt` (136 MB) next to it, so the folder must be writable; later
runs load the prototypes directly. To keep the bank elsewhere, download to `<dir>` and set `SEEDCUT_BANK=<dir>`.

The model (`Qwen/Qwen2.5-VL-7B-Instruct`) is fetched from the Hugging Face hub; set `SEEDCUT_MODEL` or pass
`--model` to use a local copy.

## Run

```bash
python run.py --image img.jpg --question "What color is the umbrella?" --options red blue green yellow
```

`--max-pixels 3411968` runs at 3.4 MP; above 1 MP the uncertainty backward uses gradient checkpointing.

## Acknowledgements

The uncertainty maps follow entropy-gradient grounding (arXiv:2604.08456). The word bank and patch scoring
follow LatentLens (MIT License). The backbone is Qwen2.5-VL-7B-Instruct.

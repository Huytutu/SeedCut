"""LatentLens evidence: question phrases, their word prototypes, and per-patch z-scores.

A patch's evidence for word w is its cosine to w's prototype, z-scored against the patch's cosines to every
prototype in the bank, so patches close to everything get no credit.
"""
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import build_inputs, grid_shape, image_positions
from .uncertainty import find_text_model

ROOT = Path(__file__).resolve().parents[1]
# generic, attribute and scene words that name no localisable object
GENERIC = set("color colour material shape size position side direction number text type kind posture state picture "
              "image photo camera viewer left right top bottom front back middle center centre part area relation "
              "relationship scene foreground background location edge group total perspective viewpoint individual "
              "question option way distance end corner row object objects thing item one".split())
_protos, _nlp = {}, []


def bank_root():
    """SEEDCUT_BANK, else data/latentlens/<backbone> picked from SEEDCUT_MODEL."""
    if os.environ.get("SEEDCUT_BANK"):
        return Path(os.environ["SEEDCUT_BANK"])
    m = os.environ.get("SEEDCUT_MODEL", "").lower()
    name = ("llava-1.6-mistral-7b" if "1.6" in m or "next" in m else
            "internvl3-8b" if "internvl" in m else "qwen2.5-vl-7b")
    return ROOT / "data/latentlens" / name


def prototypes(layer):
    """{words, index, vecs}: per word, the normalised mean of its normalised bank embeddings. Built from
    layer_<L>/embeddings_cache.pt on first use and saved as layer_<L>/prototypes.pt."""
    if layer not in _protos:
        root = bank_root()
        path = root / f"layer_{layer}/prototypes.pt"
        if not path.exists():
            cache = torch.load(root / f"layer_{layer}/embeddings_cache.pt", map_location="cpu",
                               weights_only=False, mmap=True)
            keys = [m["token_str"].strip().lower() if m["token_str"].startswith(" ") else ""
                    for m in cache["metadata"]]
            words = sorted({k for k in keys if k.isalpha() and len(k) >= 2})
            index = {w: i for i, w in enumerate(words)}
            rows = torch.tensor([index.get(k, -1) for k in keys])
            total = torch.zeros(len(words), cache["embeddings"].shape[1])
            for s in range(0, len(keys), 50000):
                r, e = rows[s:s + 50000], F.normalize(cache["embeddings"][s:s + 50000].float(), dim=-1)
                total.index_add_(0, r[r >= 0], e[r >= 0])
            torch.save({"words": words, "vecs": F.normalize(total, dim=-1).half()}, path)
        p = torch.load(path, weights_only=False)
        _protos[layer] = {"words": p["words"], "index": {w: i for i, w in enumerate(p["words"])}, "vecs": p["vecs"]}
    return _protos[layer]


def nlp():
    if not _nlp:
        import spacy
        _nlp.append(spacy.load("en_core_web_sm"))
    return _nlp[0]


def targets(question):
    """Noun-chunk roots that name an object (noun or proper noun, >= 3 letters, not generic), one per root word."""
    out = []
    for chunk in nlp()(question).noun_chunks:
        head = chunk.root
        word = head.text.lower()
        if (head.pos_ not in ("NOUN", "PROPN") or word in GENERIC or head.lemma_.lower() in GENERIC
                or not word.isalpha() or len(word) < 3):
            continue
        if all(t["word"] != word for t in out):
            out.append({"word": word, "lemma": head.lemma_.lower(), "span": (head.idx, head.idx + len(head.text))})
    return out


def patch_hidden(model, processor, image, layer):
    """Image tokens' hidden_states[layer] of a forward on the image alone -> ([n, d], grid)."""
    inputs = build_inputs(processor, [image], "", device=model.device)
    pos = image_positions(model, inputs)
    out = {}

    def grab(m, a, o):
        x = o[0] if isinstance(o, tuple) else o
        out["h"] = x[0, pos.to(x.device)].float()
    h = find_text_model(model).layers[layer - 1].register_forward_hook(grab)
    try:
        with torch.inference_mode():
            model(**inputs, logits_to_keep=1)
    finally:
        h.remove()
    return out["h"], grid_shape(model, inputs)


def word_vectors(words, layer):
    """Each word's prototype (by lemma, else surface form), or None."""
    p = prototypes(layer)
    return [p["vecs"][p["index"][w["lemma"]]].float() if w["lemma"] in p["index"] else
            p["vecs"][p["index"][w["word"]]].float() if w["word"] in p["index"] else None for w in words]


def zscores(hidden, vecs, protos):
    """hidden [n, d], vecs [T, d] -> z [n, T], each patch standardised over the whole prototype bank."""
    h = F.normalize(hidden.float(), dim=-1)
    sims = h @ protos.to(h.device, torch.float32).T
    mu, sd = sims.mean(1), sims.std(1)
    v = F.normalize(vecs.to(h.device, torch.float32), dim=-1)
    return (((h @ v.T) - mu[:, None]) / sd[:, None]).cpu()


def word_maps(hidden, grid, vecs, layer):
    """-> z [T, gh, gw], NaN for words without a prototype."""
    gh, gw = grid
    z = torch.full((len(vecs), gh, gw), float("nan"))
    ok = [i for i, v in enumerate(vecs) if v is not None]
    if ok:
        z[ok] = zscores(hidden, torch.stack([vecs[i] for i in ok]), prototypes(layer)["vecs"]).T.reshape(len(ok), gh, gw)
    return z


def cells_box(cells, grid, size):
    """Pixel box (x0, y0, x1, y1) of grid cells [(row, col)] on an image of `size`."""
    (gh, gw), (W, H) = grid, size
    sx, sy = W / gw, H / gh
    return (min(c for _, c in cells) * sx, min(r for r, _ in cells) * sy,
            (max(c for _, c in cells) + 1) * sx, (max(r for r, _ in cells) + 1) * sy)

"""LatentLens: a patch's evidence for a question noun is its cosine to the noun's contextual word prototype,
z-scored against the patch's cosines to every prototype in the bank, so hub patches get no credit."""
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import build_inputs, grid_shape, image_positions

# the bank must come from the backbone's own LLM; layer_<L>/prototypes.pt or layer_<L>/embeddings_cache.pt
BANK_ROOT = Path(os.environ.get("SEEDCUT_BANK", Path(__file__).resolve().parents[1] / "data/latentlens/qwen2.5-vl-7b"))
# attribute words that name no localisable object
ATTR = set("color colour material shape size position side direction number text type kind posture state picture "
           "image photo camera viewer left right top bottom front back middle center centre part area relation "
           "relationship scene foreground background location edge group total perspective viewpoint individual "
           "question option way distance end corner row".split())
GENERIC = ATTR | {"object", "objects", "thing", "item", "one"}
_protos, _nlp = {}, []


def prototypes(layer):
    if layer not in _protos:
        path = BANK_ROOT / f"layer_{layer}/prototypes.pt"
        if not path.exists():
            # average the normalised contextual embeddings of each whole word into one prototype
            cache = torch.load(BANK_ROOT / f"layer_{layer}/embeddings_cache.pt", map_location="cpu",
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


def targets(question):
    """Noun-chunk heads of the question that name an object."""
    if not _nlp:
        import spacy
        _nlp.append(spacy.load("en_core_web_sm"))
    out = []
    for chunk in _nlp[0](question).noun_chunks:
        head = chunk.root
        word = head.text.lower()
        if (head.pos_ not in ("NOUN", "PROPN") or word in GENERIC or head.lemma_.lower() in GENERIC
                or not word.isalpha() or len(word) < 3):
            continue
        if all(t["word"] != word for t in out):
            out.append({"word": word, "lemma": head.lemma_.lower()})
    return out


def patch_hidden(model, processor, image, layers):
    # image tokens precede the text, so their states do not depend on the question
    inputs = build_inputs(processor, [image], "", device=model.device)
    with torch.inference_mode():
        hs = model(**inputs, output_hidden_states=True).hidden_states
    pos = image_positions(model, inputs)
    return {L: hs[L][0, pos].float() for L in layers}, grid_shape(model, inputs)


def target_vectors(tgts, layer):
    """-> [T] bank prototype per target (lemma first, then surface form), None when the bank lacks it."""
    p = prototypes(layer)
    vec = [p["vecs"][p["index"][t["lemma"]]] if t["lemma"] in p["index"] else
           p["vecs"][p["index"][t["word"]]] if t["word"] in p["index"] else None for t in tgts]
    return [None if v is None else v.float() for v in vec]


def zscores(hidden, vecs, protos):
    """hidden [n, d], vecs [..., d] -> z [n, ...], each patch standardised over the whole prototype vocabulary."""
    h = F.normalize(hidden.float(), dim=-1)
    sims = h @ protos.to(h.device, torch.float32).T
    mu, sd = sims.mean(1), sims.std(1)
    flat = F.normalize(vecs.reshape(-1, vecs.shape[-1]).to(h.device, torch.float32), dim=-1)
    z = ((h @ flat.T) - mu[:, None]) / sd[:, None]
    return z.reshape(h.shape[0], *vecs.shape[:-1]).cpu()


def score_view(hidden, grid, vecs, protos):
    """-> z [T, gh, gw], NaN rows for targets without a vector."""
    gh, gw = grid
    z = torch.full((len(vecs), gh, gw), float("nan"))
    ok = [i for i, v in enumerate(vecs) if v is not None]
    if ok:
        z[ok] = zscores(hidden, torch.stack([vecs[i] for i in ok]), protos).T.reshape(len(ok), gh, gw)
    return z


def cells_box(cells, grid, frame):
    gh, gw = grid
    x0, y0, x1, y1 = frame
    sx, sy = (x1 - x0) / gw, (y1 - y0) / gh
    return (x0 + min(c for _, c in cells) * sx, y0 + min(r for r, _ in cells) * sy,
            x0 + (max(c for _, c in cells) + 1) * sx, y0 + (max(r for r, _ in cells) + 1) * sy)

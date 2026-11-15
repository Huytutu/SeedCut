"""Seed and Cut: seeds on the smoothed uncertainty map, each seed's half-maximum region, ranked top five.
Select: each question noun phrase scores the regions by the LatentLens evidence of its weakest word."""
import torch
import torch.nn.functional as F

from . import semantic as S
from .uncertainty import components


def seeds(A, m=20, r=2):
    """Up to m greedy maxima of A, more than r cells apart (Chebyshev)."""
    gh, gw = A.shape
    out = []
    for i in A.flatten().argsort(descending=True).tolist():
        y, x = divmod(i, gw)
        if all(max(abs(y - a), abs(x - b)) > r for a, b in out):
            out.append((y, x))
            if len(out) == m:
                break
    return out


def propose(U, k=5, m=20, t=.5):
    """U [gh, gw] -> top-k [(score, mask, [density, mass], seed)]."""
    Uloc = F.avg_pool2d(U[None, None], 3, 1, 1, count_include_pad=False)[0, 0]
    tot = float(U.sum())
    regs = []
    for s in seeds(Uloc, m):
        # the 4-connected component around s of the cells reaching t times the seed's own smoothed value
        comp = next(c for _, c in components(Uloc >= t * Uloc[s], Uloc) if s in c)
        mk = torch.zeros_like(U, dtype=torch.bool)
        for r, c in comp:
            mk[r, c] = True
        if not any(bool((mk == r).all()) for r, _ in regs):
            regs.append((mk, s))
    raw = torch.tensor([[float(U[mk].mean()), float(U[mk].sum()) / tot] for mk, _ in regs])
    score = raw[:, 0] * raw[:, 1].sqrt()  # density * sqrt(mass)
    keep = []
    for i in score.argsort(descending=True).tolist():
        mk = regs[i][0]
        # components of level sets of one map are nested or disjoint: any shared cell means a nested copy
        if not any(bool((mk & regs[j][0]).any()) for j in keep):
            keep.append(i)
        if len(keep) == k:
            break
    return [(float(score[i]), regs[i][0], raw[i].tolist(), regs[i][1]) for i in keep]


def word_scores(words, hidden, grid, props, layer):
    """Evidence per (word, region): the mean of the region's top-3 z. -> [T, R], or None."""
    z = S.word_maps(hidden, grid, S.word_vectors(words, layer), layer)
    ok = [i for i in range(len(words)) if not torch.isnan(z[i]).all()]
    if not ok:
        return None
    out = torch.zeros(len(ok), len(props))
    for j, (_, mk, _, _) in enumerate(props):
        cells = mk.nonzero()
        q = min(3, len(cells))
        for a, i in enumerate(ok):
            out[a, j] = float(z[i][cells[:, 0], cells[:, 1]].topk(q).values.mean())
    return out


def phrase_scores(text, hidden, grid, props, layer):
    """-> ([P, R] per noun phrase: the min evidence over its root and amod/compound modifiers, the phrases as word
    lists), or (None, []) when no phrase word has a prototype."""
    tgts = S.targets(text)
    doc = {t.idx: t for t in S.nlp()(text)} if tgts else {}
    idx, phrases = S.prototypes(layer)["index"], []
    for tg in tgts:
        root = doc[tg["span"][0]]
        toks = [root] + [c for c in root.children
                         if c.dep_ in ("amod", "compound") and c.text.isalpha() and len(c.text) >= 3]
        ws = [{"word": c.text.lower(), "lemma": c.lemma_.lower()} for c in toks]
        ws = [w for w in ws if w["lemma"] in idx or w["word"] in idx]
        if ws:
            phrases.append(ws)
    if not phrases:
        return None, []
    Mw = word_scores([w for ws in phrases for w in ws], hidden, grid, props, layer)
    rows, i = [], 0
    for ws in phrases:
        rows.append(Mw[i:i + len(ws)].min(0).values)
        i += len(ws)
    return torch.stack(rows), [[w["word"] for w in ws] for ws in phrases]

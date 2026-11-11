"""SeedCut regions: uncertainty-relative seeds, a seeded normalized cut per seed, S-ranked top five, then a
size-normalised LatentLens round-robin over the question nouns picks two.
"""
import torch
import torch.nn.functional as F

from . import semantic as S
from .boxes import clip_box
from .egg import components

# ViCrop's generic prompt: U0 is the map a question-free prompt already produces
GENERIC = "Write a general description of the image."
# 8-neighbour offsets, each undirected edge once
OFFS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def u_rel(U, U0):
    # 1/|Omega| keeps cells the generic prompt ignores from blowing up
    return (U / U.sum()) / (U0 / U0.sum() + 1 / U0.numel())


def seeds(A, m=20, r=2):
    gh, gw = A.shape
    out = []
    for i in A.flatten().argsort(descending=True).tolist():
        y, x = divmod(i, gw)
        if all(max(abs(y - a), abs(x - b)) > r for a, b in out):
            out.append((y, x))
            if len(out) == m:
                break
    return out


def edge_weights(Z, tau=0.1):
    """Z [gh, gw, d] centred -> {offset: [gh, gw] weight of the edge (y,x)-(y+dy,x+dx)}, 0 where it leaves the grid."""
    Zn = F.normalize(Z, dim=-1)
    gh, gw = Z.shape[:2]
    W = {}
    for dy, dx in OFFS:
        w = torch.zeros(gh, gw)
        y0, y1 = 0, gh - dy
        x0, x1 = max(0, -dx), gw - max(0, dx)
        cos = (Zn[y0:y1, x0:x1] * Zn[y0 + dy:y1 + dy, x0 + dx:x1 + dx]).sum(-1)
        w[y0:y1, x0:x1] = torch.exp(cos / tau)
        W[dy, dx] = w
    return W


def seeded_cut(W, seed, half):
    """Normalized cut of the window around seed; -> bool [gh, gw] mask of the seed's side, seed's component only."""
    gh, gw = W[0, 1].shape
    y, x = seed
    ya, yb, xa, xb = max(0, y - half), min(gh, y + half + 1), max(0, x - half), min(gw, x + half + 1)
    h, w = yb - ya, xb - xa
    n = h * w
    A = torch.zeros(n, n, dtype=torch.float64)
    idx = torch.arange(n).reshape(h, w)
    for (dy, dx), wt in W.items():
        r0, r1, c0, c1 = 0, h - dy, max(0, -dx), w - max(0, dx)
        a, b = idx[r0:r1, c0:c1].flatten(), idx[r0 + dy:r1 + dy, c0 + dx:c1 + dx].flatten()
        v = wt[ya + r0:ya + r1, xa + c0:xa + c1].flatten().double()
        A[a, b] = v
        A[b, a] = v
    d = A.sum(1)
    Dm = d.rsqrt()
    L = torch.eye(n, dtype=torch.float64) - Dm[:, None] * A * Dm[None]
    y2 = (Dm * torch.linalg.eigh(L).eigenvectors[:, 1]).reshape(h, w)
    side = (y2 > y2.mean()) if y2[y - ya, x - xa] > y2.mean() else (y2 <= y2.mean())
    comp = next(cells for _, cells in components(side, y2.abs()) if (y - ya, x - xa) in cells)
    mask = torch.zeros(gh, gw, dtype=torch.bool)
    for r, c in comp:
        mask[ya + r, xa + c] = True
    return mask


def coherence(W, mask):
    """Mean affinity of edges inside R vs mean affinity of edges cut by R's boundary, as m_in / (m_in + m_out).
    Means, not sums: summed E_in grows with area and E_out with perimeter, so the ratio of sums tracked size."""
    e_in = e_out = n_in = n_out = 0.0
    m = mask.float()
    gh, gw = mask.shape
    for (dy, dx), wt in W.items():
        # partner of (y,x) is (y+dy, x+dx); shift the mask so both ends line up
        p = torch.zeros_like(m)
        y1, xa, xb = gh - dy, max(0, -dx), gw - max(0, dx)
        p[:y1, xa:xb] = m[dy:, xa + dx:xb + dx]
        ok = (wt > 0).float()  # edges that leave the grid have weight 0 and do not exist
        cut = m * (1 - p) + (1 - m) * p
        e_in, n_in = e_in + float((wt * m * p).sum()), n_in + float((ok * m * p).sum())
        e_out, n_out = e_out + float((wt * cut).sum()), n_out + float((ok * cut).sum())
    m_in, m_out = e_in / max(n_in, 1), e_out / max(n_out, 1)
    return m_in / (m_in + m_out + 1e-9)


def propose(U, Z, k=5, m=20, fracs=(0.06, 0.12, 0.24), nms=0.8):
    """U [gh, gw] relative uncertainty, Z [gh, gw, d] patch states -> top-k [(score, mask, [density, mass, coh], seed)]."""
    Uloc = F.avg_pool2d(U[None, None], 3, 1, 1, count_include_pad=False)[0, 0]
    sd = seeds(Uloc, m)
    W = edge_weights(Z - Z.reshape(-1, Z.shape[-1]).mean(0))
    gh, gw = U.shape
    tot = float(U.sum())

    def parts(mk):
        return [float(U[mk].mean()), float(U[mk].sum()) / tot, coherence(W, mk)]

    regs = []
    for s in sd:
        cand = [seeded_cut(W, s, max(2, round(f * max(gh, gw)) // 2)) for f in fracs]
        # characteristic scale: the window whose region scores best, so each seed yields one region
        sc = [(lambda p: p[0] * p[1] ** .5 * p[2])(parts(mk)) for mk in cand]
        mk = cand[max(range(len(cand)), key=sc.__getitem__)]
        if not any(bool((mk == r).all()) for r, _ in regs):
            regs.append((mk, s))
    raw = torch.tensor([parts(mk) for mk, _ in regs])
    # product form: mass enters as a square root, so a large region cannot outrank a dense one on size alone
    score = raw[:, 0] * raw[:, 1].sqrt() * raw[:, 2]
    keep = []
    for i in score.argsort(descending=True).tolist():
        mk = regs[i][0]
        # overlap over the SMALLER mask, not IoU: a region nested in a kept one is the same evidence at another
        # scale, yet their IoU is low, so IoU-NMS let one object fill all five slots
        if all(float((mk & regs[j][0]).sum()) / min(float(mk.sum()), float(regs[j][0].sum())) < nms for j in keep):
            keep.append(i)
        if len(keep) == k:
            break
    return [(float(score[i]), regs[i][0], raw[i].tolist(), regs[i][1]) for i in keep]


def mask_box(mask, size):
    cells = mask.nonzero().tolist()
    return clip_box(S.cells_box([tuple(c) for c in cells], mask.shape, (0, 0, *size)), size)


def noun_scores(text, hidden, grid, props, K=200, layer=24):
    """Size-normalised evidence per (noun of `text`, region): each region's mean top-3 LatentLens z is standardised
    against the same statistic on K random translations of the SAME mask, so a larger region gains nothing from
    holding more cells. -> [T, R] over the nouns with a bank vector, or None."""
    tgts = S.targets(text)
    if not tgts:
        return None
    z = S.score_view(hidden, grid, S.target_vectors(tgts, layer), S.prototypes(layer)["vecs"])
    ok = [i for i in range(len(tgts)) if not torch.isnan(z[i]).all()]
    if not ok:
        return None
    gh, gw = grid
    g = torch.Generator().manual_seed(0)
    out = torch.zeros(len(ok), len(props))
    for j, (_, mk, _, _) in enumerate(props):
        cells = mk.nonzero()
        rel = cells - cells.min(0).values
        h, w = int(rel[:, 0].max()) + 1, int(rel[:, 1].max()) + 1
        oy = torch.randint(0, gh - h + 1, (K,), generator=g)
        ox = torch.randint(0, gw - w + 1, (K,), generator=g)
        ys, xs = rel[None, :, 0] + oy[:, None], rel[None, :, 1] + ox[:, None]
        k = min(3, len(cells))
        for a, i in enumerate(ok):
            t = float(z[i][cells[:, 0], cells[:, 1]].topk(k).values.mean())
            null = z[i][ys, xs].topk(k, dim=1).values.mean(1)
            out[a, j] = (t - float(null.mean())) / (float(null.std()) + 1e-6)
    return out


def round_robin_idx(M, k=2):
    # each noun's best region before any noun's second, so a two-object question gets one region per object
    order = [M[a].argsort(descending=True).tolist() for a in range(M.shape[0])]
    picks = []
    for depth in range(M.shape[1]):
        for o in order:
            if o[depth] not in picks and len(picks) < k:
                picks.append(o[depth])
    return picks

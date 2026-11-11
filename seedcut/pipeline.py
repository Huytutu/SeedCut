"""SeedCut: relative-uncertainty seeds -> seeded normalized cuts -> noun round-robin -> one combined re-look answer."""
import torch

from . import model as M
from . import semantic as S
from .boxes import area_frac, crop_ok, pad_box
from .egg import entropy_saliency
from .propose import GENERIC, mask_box, noun_scores, propose, round_robin_idx, u_rel

LAYER = 24


def select_regions(model, processor, image, question, bare_question):
    """-> (picked indices, picked boxes in pixels); ([], []) when a backward fails."""
    U = entropy_saliency(model, processor, image, question, M.SHORT)
    # U0 comes from a question-free prompt: dividing by it keeps only the uncertainty THIS question adds
    U0 = entropy_saliency(model, processor, image, GENERIC, M.SHORT) if U is not None else None
    if U0 is None:
        return [], []
    hidden, grid = S.patch_hidden(model, processor, image, [LAYER])
    assert tuple(grid) == tuple(U.shape), (grid, U.shape)
    props = propose(u_rel(U.float(), U0.float()), hidden[LAYER].cpu().reshape(*grid, -1))
    cands = [mask_box(mk, image.size) for _, mk, _, _ in props]
    Mn = noun_scores(bare_question, hidden[LAYER], grid, props, layer=LAYER)
    # no question noun with a bank vector: our own best region alone
    ids = [0] if Mn is None else round_robin_idx(Mn)
    return ids, [cands[j] for j in ids]


def run_record(model, processor, rec, pad=0.15, max_new_tokens=512, suffix=M.COT, extract=None):
    """rec: {id, image (PIL), question (with options), bare_question, letters, gt (optional)}."""
    image, question = rec["image"], rec["question"]
    letters = rec.get("letters") or "ABCD"
    gen = {"last": letters[-1], "suffix": suffix, "extract": extract, "max_new_tokens": max_new_tokens}
    r = {"case": rec.get("id"), "gt": rec.get("gt")}

    ids, regions = select_regions(model, processor, image, question, rec.get("bare_question", question))
    box = (min(b[0] for b in regions), min(b[1] for b in regions),
           max(b[2] for b in regions), max(b[3] for b in regions)) if regions else None
    r.update(picks=ids, box=None if box is None else [round(v, 1) for v in box],
             regions=[[round(v, 1) for v in b] for b in regions],
             area_frac=None if box is None else round(area_frac(pad_box(box, pad, image.size), image.size), 4))

    # one generation sees the full image, the enclosing crop of the picks, and each pick's crop
    pred, text = None, ""
    if box is not None:
        crops = [crop_ok(image, pad_box(b, pad, image.size)) for b in [box] + regions]
        if all(c is not None for c in crops):
            pred, text = M.answer(model, processor, [image] + crops, question, **gen)
    r.update(relook_pred=pred, relook_text=text)

    # an unparsed answer, a degenerate crop or a failed backward keeps the base answer
    base_pred, base_text = (None, "") if pred is not None else M.answer(model, processor, [image], question, **gen)
    r.update(base_pred=base_pred, base_text=base_text)
    r["final_pred"] = pred if pred is not None else base_pred
    if r["gt"] is not None:
        r["final_correct"] = r["final_pred"] == r["gt"]
    return r

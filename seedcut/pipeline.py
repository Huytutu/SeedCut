"""SeedCut: uncertainty seeds -> half-maximum regions -> phrase Select -> one answer on [image, masked view]."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import model as M
from . import semantic as S
from .propose import phrase_scores, propose
from .uncertainty import entropy_saliency

LAYER = 24


def select(model, processor, image, question, bare_question):
    """-> (picked region ids, picked masks [gh, gw] OR-ed, pixel boxes, phrases), or None when U fails."""
    # Qwen: the patch states come from U's own forward; the other backbones run one more forward
    cap = {"layer": LAYER} if getattr(processor, "backend", None) is None else None
    U = entropy_saliency(model, processor, image, question, M.SHORT, capture=cap)
    if U is None:
        return None
    hidden, grid = (cap["h"], tuple(U.shape)) if cap and "h" in cap else S.patch_hidden(model, processor, image, LAYER)
    assert tuple(grid) == tuple(U.shape), (grid, U.shape)
    props = propose(U.float())
    Mp, phrases = phrase_scores(bare_question, hidden, grid, props, LAYER)
    # R1 always (also when no phrase word has a prototype), plus each phrase's best region
    ids = sorted({0} | ({int(Mp[a].argmax()) for a in range(Mp.shape[0])} if Mp is not None else set()))
    boxes = [S.cells_box(props[j][1].nonzero().tolist(), grid, image.size) for j in ids]
    return ids, torch.stack([props[j][1] for j in ids]).any(0), boxes, phrases


def masked_view(image, keep):
    """The image black outside the kept cells (plus one cell of margin), cropped to their box -> (view, box)."""
    m = F.interpolate(F.max_pool2d(keep.float()[None, None], 3, 1, 1), size=(image.height, image.width),
                      mode="nearest")[0, 0].numpy() > 0
    ys, xs = m.nonzero()
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    arr = np.asarray(image)
    view = Image.fromarray(arr * m[..., None].astype(arr.dtype)).crop(box)
    return (view if min(view.size) >= 8 else None), box


def run_record(model, processor, rec, suffix, extract, max_new_tokens=32):
    """rec: {image (PIL RGB), question (with options, as prompted), bare_question (without options)}.
    suffix / extract: the benchmark's answer instruction and letter parser."""
    image, question = rec["image"], rec["question"]
    gen = {"suffix": suffix, "extract": extract, "max_new_tokens": max_new_tokens}
    r = {"picks": None, "phrases": [], "regions": [], "box": None}
    sel = select(model, processor, image, question, rec.get("bare_question", question))
    view = None
    if sel is not None:
        ids, keep, boxes, phrases = sel
        view, box = masked_view(image, keep)
        r.update(picks=ids, phrases=phrases, regions=[[round(v, 1) for v in b] for b in boxes], box=list(box))
    pred, text = M.answer(model, processor, [image, view], question, **gen) if view is not None else (None, "")
    r.update(look_pred=pred, look_text=text)
    # an unparsed answer, a degenerate view or a failed backward keeps the full-image answer
    if pred is None:
        base_pred, base_text = M.answer(model, processor, [image], question, **gen)
        r.update(base_pred=base_pred, base_text=base_text)
    r["final_pred"] = pred if pred is not None else r["base_pred"]
    return r

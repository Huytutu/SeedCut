"""Entropy-Gradient Grounding (arXiv:2604.08456): the answer entropy backpropagated to the projected visual
embeddings gives a per-patch uncertainty map."""
import torch

from . import model as M
from .model import build_inputs, grid_shape, image_positions


def find_text_model(model):
    # the text model's inputs_embeds holds the merged visual tokens: those are the embeddings EGG differentiates
    hits = [n for n, _ in model.named_modules() if n.endswith("layers.0") and "visual" not in n and "vision" not in n]
    assert len(hits) == 1, f"expected one text layers.0, got {hits}"
    return model.get_submodule(hits[0].rsplit(".layers.0", 1)[0])


def entropy_saliency(model, processor, image, question, suffix, append="<answer>"):
    """-> [gh, gw] saliency, or None on OOM. One forward + one backward, no generation."""
    inputs = build_inputs(processor, [image], question, suffix=suffix, device=model.device, append=append)
    store = {}

    def pre_hook(mod, a, kw):
        e = kw["inputs_embeds"].detach().requires_grad_(True)
        store["e"] = e
        kw["inputs_embeds"] = e
        return a, kw

    # above the 1 MP default the backward only fits a 24 GB card with the decoder checkpointed
    ckpt = M.MAX_PIXELS > 1280 * 28 * 28
    h = find_text_model(model).register_forward_pre_hook(pre_hook, with_kwargs=True)
    if ckpt:
        torch.cuda.empty_cache()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    try:
        model.zero_grad(set_to_none=True)
        p = torch.softmax(model(**inputs, use_cache=False).logits[0, -1].float(), dim=-1)
        (-(p * torch.log(p + 1e-12)).sum()).backward()
    except torch.cuda.OutOfMemoryError:
        store.pop("e", None)
        torch.cuda.empty_cache()
        return None
    finally:
        h.remove()
        if ckpt:
            model.eval()
            model.gradient_checkpointing_disable()

    g = store["e"].grad if "e" in store else None
    if g is None:
        return None
    sal = g[0, image_positions(model, inputs)].detach().float().norm(dim=-1).cpu()
    gh, gw = grid_shape(model, inputs)
    return sal.reshape(gh, gw) if sal.numel() == gh * gw else None


def components(mask, sal):
    """4-connected blobs of a boolean mask, ranked by accumulated saliency."""
    gh, gw = mask.shape
    seen = torch.zeros_like(mask)
    out = []
    for r in range(gh):
        for c in range(gw):
            if mask[r, c] and not seen[r, c]:
                stack, comp = [(r, c)], []
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < gh and 0 <= nx < gw and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                out.append((sum(float(sal[y, x]) for y, x in comp), comp))
    return sorted(out, key=lambda t: -t[0])

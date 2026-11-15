"""U: the entropy of the answer distribution, backpropagated to the projected visual embeddings, as one norm per
visual token. One forward and one backward, no generation; the forward also yields the patch states Select reads.
"""
import torch

from . import model as M
from .model import build_inputs, grid_shape, image_positions


def find_text_model(model):
    # the text model's inputs_embeds is the merged sequence: its image positions are the projected visual embeddings.
    # The leaf sits outside the decoder layers, since under gradient checkpointing a leaf made inside a layer is
    # replaced by the recompute and never gets a gradient
    hits = [n for n, _ in model.named_modules() if n.endswith("layers.0") and "visual" not in n and "vision" not in n]
    assert len(hits) == 1, f"expected one text layers.0, got {hits}"
    return model.get_submodule(hits[0].rsplit(".layers.0", 1)[0])


def entropy_saliency(model, processor, image, question, suffix, append="<answer>", capture=None):
    """-> [gh, gw] uncertainty map, or None when the backward does not fit.
    capture={"layer": L} also returns, in capture["h"], the image tokens' hidden_states[L] of this same forward."""
    inputs = build_inputs(processor, [image], question, suffix=suffix, device=model.device, append=append)
    store = {}
    pos = image_positions(model, inputs)

    def pre_hook(mod, a, kw):
        # swap the decoder's input for a leaf that requires grad, so nothing backprops into the vision tower
        e = kw["inputs_embeds"].detach().requires_grad_(True)
        store["e"] = e
        kw["inputs_embeds"] = e
        return a, kw

    # above 1 MP (and for InternVL3's tiles and LLaVA-1.6's anyres grid) the backward needs gradient checkpointing
    ckpt = M.MAX_PIXELS > 1280 * 28 * 28 or getattr(processor, "backend", None) in ("internvl", "llava_next")
    hooks = [find_text_model(model).register_forward_pre_hook(pre_hook, with_kwargs=True)]
    if capture is not None:
        # hidden_states[L] is the output of decoder layer L-1; keep the first call, not checkpointing's recompute.
        # Image tokens precede the question, so these are the question-free states
        def grab(mod, a, o):
            x = o[0] if isinstance(o, tuple) else o
            capture.setdefault("raw", x[0, pos.to(x.device)].detach())
        hooks.append(find_text_model(model).layers[capture["layer"] - 1].register_forward_hook(grab))
    if ckpt:
        torch.cuda.empty_cache()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()  # HF checkpoints only in training mode
    try:
        model.zero_grad(set_to_none=True)
        logits = model(**inputs, use_cache=False).logits[0, -1]
        p = torch.softmax(logits.float(), dim=-1)
        (-(p * torch.log(p + 1e-12)).sum()).backward()
    except torch.cuda.OutOfMemoryError:
        store.pop("e", None)
        torch.cuda.empty_cache()
        return None
    finally:
        for h in hooks:
            h.remove()
        if ckpt:
            model.eval()
            model.gradient_checkpointing_disable()

    g = store["e"].grad if "e" in store else None
    if g is None:
        return None
    if capture is not None and "raw" in capture:
        capture["h"] = capture.pop("raw").float()
    sal = g[0, pos].detach().float().norm(dim=-1).cpu()
    gh, gw = grid_shape(model, inputs)
    if sal.numel() != gh * gw:
        return None
    return sal.reshape(gh, gw)


def components(mask, sal):
    """4-connected blobs of a boolean map, ranked by accumulated `sal`."""
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

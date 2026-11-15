"""LLaVA-1.6 (LLaVA-NeXT, Mistral-7B) backend for SeedCut (llava-hf weights, its own chat template).

anyres: each image is a 576-token 336px base view followed by the unpadded high-resolution tile grid, one newline
token closing each row. The uncertainty map and LatentLens read the high-resolution grid only; after unpadding it spans the original
image exactly, so boxes map back with the plain (0, 0, W, H) frame.
"""
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from transformers.image_processing_utils import select_best_resolution


def load_llava_next(path, **kw):
    model = LlavaNextForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16, **kw)
    processor = AutoProcessor.from_pretrained(path)
    processor.backend = "llava_next"
    return model, processor


def build_inputs(processor, images, question, suffix="", device=None, append=""):
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question + suffix}]
    chat = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True) + append
    inputs = processor(images=images, text=chat, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].half()
    return inputs.to(device) if device is not None else inputs


def hires_grid(model, inputs):
    # the processor's own feature count (LlavaNextProcessor._get_unpadded_features), for image 0
    side = model.config.vision_config.image_size // model.config.vision_config.patch_size
    h, w = inputs["image_sizes"][0].tolist()
    bh, bw = select_best_resolution([h, w], model.config.image_grid_pinpoints)
    ch, cw = side * (bh // model.config.vision_config.image_size), side * (bw // model.config.vision_config.image_size)
    if w / h > cw / ch:
        ch -= ((ch - int(round(h * (cw / w), 7))) // 2) * 2
    else:
        cw -= ((cw - int(round(w * (ch / h), 7))) // 2) * 2
    return side * side, ch, cw


def image_positions(model, inputs):
    idx = (inputs.input_ids[0] == model.config.image_token_id).nonzero(as_tuple=True)[0]
    base, ch, cw = hires_grid(model, inputs)
    # inputs hold one image here (uncertainty, patch_hidden); its tokens are base + ch rows of (cw + newline)
    assert len(idx) == base + ch * (cw + 1), (len(idx), base, ch, cw)
    return idx[base:].reshape(ch, cw + 1)[:, :cw].flatten()


def grid_shape(model, inputs):
    _, ch, cw = hires_grid(model, inputs)
    return ch, cw

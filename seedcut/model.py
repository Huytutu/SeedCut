"""Backbone loading and the forward primitives every stage is built from (Qwen2.5-VL; InternVL3 and LLaVA-1.6 in
their own modules)."""
import os
import random

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28  # Qwen2.5-VL's largest input, 12.8 MP

# the uncertainty probe: the next token after "<answer>" is the answer letter
SHORT = ("\nSelect the best answer to the above multiple-choice question based on the image. "
         "Respond with only the letter of the correct option between <answer> and </answer>.")


def set_deterministic(seed=0):
    # CUBLAS_WORKSPACE_CONFIG must be set before the CUDA context exists
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(path):
    # on a GPU pair, keep few layers on cuda:0 so the uncertainty backward at full resolution fits
    n = torch.cuda.device_count()
    kw = {"device_map": "cuda:0"} if n == 1 else {"device_map": "auto",
                                                  "max_memory": {0: "5GiB", **{i: "22GiB" for i in range(1, n)}}}
    if "internvl" in path.lower():
        from .internvl import load_internvl
        return load_internvl(path, **kw)
    if "1.6" in path or "next" in path.lower():
        from .llava_next import load_llava_next
        return load_llava_next(path, **kw)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, **kw)
    return model, AutoProcessor.from_pretrained(path)


def _backend(processor):
    return getattr(processor, "backend", None)


def image_positions(model, inputs):
    if hasattr(inputs, "tile_grid"):
        from .internvl import image_positions as f
        return f(model, inputs)
    if "image_sizes" in inputs:
        from .llava_next import image_positions as f
        return f(model, inputs)
    return (inputs.input_ids[0] == model.config.image_token_id).nonzero(as_tuple=True)[0]


def grid_shape(model, inputs):
    if hasattr(inputs, "tile_grid"):
        from .internvl import grid_shape as f
        return f(model, inputs)
    if "image_sizes" in inputs:
        from .llava_next import grid_shape as f
        return f(model, inputs)
    merge = model.config.vision_config.spatial_merge_size
    _, h, w = inputs.image_grid_thw[0].tolist()
    return h // merge, w // merge


def build_inputs(processor, images, question, suffix="", device=None, append=""):
    """images: a list; the full image alone, or the full image followed by the masked view."""
    if _backend(processor) == "internvl":
        from .internvl import build_inputs as f
        return f(processor, images, question, suffix, device, append)
    if _backend(processor) == "llava_next":
        from .llava_next import build_inputs as f
        return f(processor, images, question, suffix, device, append)
    content = [{"type": "image", "image": im, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS} for im in images]
    content.append({"type": "text", "text": question + suffix})
    messages = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + append
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt")
    return inputs.to(device) if device is not None else inputs


def answer(model, processor, images, question, suffix, extract, max_new_tokens=32):
    """Greedy answer -> (letter or None, raw text)."""
    inputs = build_inputs(processor, images, question, suffix=suffix, device=model.device)
    with torch.inference_mode():
        seq = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    text = processor.tokenizer.decode(seq[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return extract(text), text

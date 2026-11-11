"""Qwen2.5-VL loading and the forward primitives every stage is built from."""
import os
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28  # raise before loading images for higher resolution (e.g. 3411968)

COT = "\n\nThink step by step and put the final answer in <answer>...</answer>."
SHORT = ("\nSelect the best answer to the above multiple-choice question based on the image. "
         "Respond with only the letter of the correct option between <answer> and </answer>.")

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def load_model(path=None):
    path = str(path or os.environ.get("SEEDCUT_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    n = torch.cuda.device_count()
    kw = {"device_map": "cuda:0"} if n == 1 else {"device_map": "auto"}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, **kw)
    return model, AutoProcessor.from_pretrained(path)


def image_positions(model, inputs):
    return (inputs.input_ids[0] == model.config.image_token_id).nonzero(as_tuple=True)[0]


def grid_shape(model, inputs):
    merge = model.config.vision_config.spatial_merge_size
    _, h, w = inputs.image_grid_thw[0].tolist()
    return h // merge, w // merge


def build_inputs(processor, images, question, suffix=COT, device=None, append=""):
    content = [{"type": "image", "image": im, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
               for im in images]
    content.append({"type": "text", "text": question + suffix})
    messages = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + append
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt")
    return inputs.to(device) if device is not None else inputs


def extract_answer(text, last="D"):
    m = _ANSWER_RE.search(text)
    if not m:
        return None
    m2 = re.match(rf"\s*([A-{last}a-{last.lower()}])\b", m.group(1).strip())
    return m2.group(1).upper() if m2 else None


def answer(model, processor, images, question, max_new_tokens=512, last="D", suffix=COT, extract=None):
    inputs = build_inputs(processor, images, question, suffix=suffix, device=model.device)
    with torch.inference_mode():
        seq = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    text = processor.tokenizer.decode(seq[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return (extract(text) if extract else extract_answer(text, last)), text

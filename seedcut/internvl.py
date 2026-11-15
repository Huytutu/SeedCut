"""InternVL3-8B backend for SeedCut (transformers-native OpenGVLab/InternVL3-8B-hf).

Dynamic tiling as in the official inference: the image is resized to cols x rows tiles of 448 px (no padding), each
tile gives 16 x 16 tokens after pixel shuffle, tiles row-major, then one thumbnail tile. The uncertainty map and LatentLens read the
tile grid only, which spans the whole original image, so boxes map back with the plain (0, 0, W, H) frame.
"""
import torch
from transformers import AutoProcessor, InternVLForConditionalGeneration
from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas

# InternVL's documented test-time maximum (InternVL-1.5: up to 40 tiles, i.e. 4K); the released default is 12
MAX_TILES = 40


def load_internvl(path, **kw):
    model = InternVLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, **kw)
    processor = AutoProcessor.from_pretrained(path)
    processor.backend = "internvl"
    return model, processor


def build_inputs(processor, images, question, suffix="", device=None, append=""):
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question + suffix}]
    chat = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True) + append
    inputs = processor(images=images, text=chat, crop_to_patches=True, max_patches=MAX_TILES, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    cols, rows = get_optimal_tiled_canvas((images[0].height, images[0].width), (448, 448), 1, MAX_TILES)
    inputs = inputs.to(device) if device is not None else inputs
    # not a model input: image 0's tile layout, for image_positions / grid_shape
    inputs.tile_grid = (rows, cols)
    return inputs


def image_positions(model, inputs):
    rows, cols = inputs.tile_grid
    side = int(model.config.image_seq_length ** 0.5)
    idx = (inputs.input_ids[0] == model.config.image_token_id).nonzero(as_tuple=True)[0]
    if rows * cols == 1:
        return idx[:side * side]
    # inputs hold one image here (uncertainty, patch_hidden): its tiles, then the thumbnail
    assert len(idx) == (rows * cols + 1) * side * side, (len(idx), rows, cols)
    return idx[:rows * cols * side * side].reshape(rows, cols, side, side).permute(0, 2, 1, 3).flatten()


def grid_shape(model, inputs):
    rows, cols = inputs.tile_grid
    side = int(model.config.image_seq_length ** 0.5)
    return rows * side, cols * side

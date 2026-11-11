"""Answer one multiple-choice question about one image with SeedCut.

python run.py --image img.jpg --question "What color is the umbrella?" --options red blue green yellow
"""
import argparse
import json

import torch
from PIL import Image

from seedcut import model as M
from seedcut import run_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--options", nargs="+", required=True)
    ap.add_argument("--model", default=None, help="Qwen2.5-VL-7B-Instruct path or hub id")
    ap.add_argument("--max-pixels", type=int, default=M.MAX_PIXELS)
    a = ap.parse_args()

    M.MAX_PIXELS = a.max_pixels
    torch.manual_seed(0)
    model, processor = M.load_model(a.model)
    model.eval()
    letters = "".join(chr(65 + i) for i in range(len(a.options)))
    question = a.question + "\n" + "\n".join(f"({l}) {o}" for l, o in zip(letters, a.options))
    rec = {"id": a.image, "image": Image.open(a.image).convert("RGB"), "question": question,
           "bare_question": a.question, "letters": letters}
    print(json.dumps(run_record(model, processor, rec), indent=2))


if __name__ == "__main__":
    main()

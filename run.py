"""Answer one multiple-choice question about one image with SeedCut.

python run.py --sample 0
python run.py --image img.jpg --question "What is the color of the umbrella?" --options red blue green yellow
"""
import argparse
import json
import os
import re
from pathlib import Path

from PIL import Image

SAMPLES = Path(__file__).resolve().parent / "examples/samples.json"


def load_sample(i):
    """A V*Bench question from examples/samples.json; its image is fetched from the craigwu/vstar_bench dataset."""
    from huggingface_hub import hf_hub_download
    s = json.loads(SAMPLES.read_text())[i]
    path = hf_hub_download("craigwu/vstar_bench", s["image"], repo_type="dataset")
    return path, s["question"], s["options"], s["answer"]


def vstar_extract(text):
    # lmms-eval's extract_answer_letter
    text = text.strip().upper()
    for p in [r"^([A-D])\s*[\.)\]]*", r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]+([A-D])",
              r"\(([A-D])\)", r"([A-D])\s*(?:\.|\)|])", r"(?:^|\s)([A-D])(?:\s|$)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    letters = re.findall(r"[A-D]", text)
    if len(letters) == 1:
        return letters[0]
    return text[0] if text and text[0] in "ABCD" else None


def hrbench_extract(text):
    # first capital A-D anywhere in the reply
    if len(text) == 1:
        return text[0]
    return next((c for c in text if c in "ABCD"), None)


# per benchmark: (options -> prompt text, answer instruction, parser(text, letters), max new tokens, min pixels)
PROMPTS = {
    "vstar": (lambda q, opts, L: q + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(L, opts)),
              "\nAnswer with the option's letter from the given choices directly.",
              lambda t, L: vstar_extract(t), 32, 256 * 28 * 28),
    "hrbench": (lambda q, opts, L: q + "\n" + "".join(f"{l}. {o}\n" for l, o in zip(L, opts)),
                "Answer the option letter directly.",
                lambda t, L: hrbench_extract(t), 256, 3136),
    "mmerw": (lambda q, opts, L: q + "\nThe choices are listed below:\n" + "\n".join(f"({l}) {o}" for l, o in zip(L, opts)),
              "Select the best answer to the above multiple-choice question based on the image. "
              "Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:",
              lambda t, L: (lambda m: m.group(1) if m else None)(re.search(rf"\b([{''.join(L)}])\b", t)), 32, 3136),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, help="run sample i of examples/samples.json instead (0, 1 or 2)")
    ap.add_argument("--image")
    ap.add_argument("--question")
    ap.add_argument("--options", nargs="+")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="Qwen2.5-VL-7B-Instruct, InternVL3-8B-hf or llava-v1.6-mistral-7b-hf (path or hub id)")
    ap.add_argument("--prompt", default="vstar", choices=list(PROMPTS), help="prompt format of the benchmark")
    ap.add_argument("--max-pixels", type=int, default=16384 * 28 * 28, help="Qwen2.5-VL input budget (12.8 MP)")
    a = ap.parse_args()
    answer = None
    if a.sample is not None:
        a.image, a.question, a.options, answer = load_sample(a.sample)
    elif not (a.image and a.question and a.options):
        ap.error("give --sample, or --image, --question and --options")

    os.environ["SEEDCUT_MODEL"] = a.model  # picks the backbone's word bank, unless SEEDCUT_BANK is set
    from seedcut import model as M
    from seedcut import run_record

    build, suffix, parse, max_new, min_pixels = PROMPTS[a.prompt]
    M.MIN_PIXELS, M.MAX_PIXELS = min_pixels, a.max_pixels
    M.set_deterministic(0)
    model, processor = M.load_model(a.model)
    model.eval()
    model.requires_grad_(False)  # the uncertainty backward needs a graph through activations only
    letters = [chr(65 + i) for i in range(len(a.options))]
    rec = {"image": Image.open(a.image).convert("RGB"), "question": build(a.question, a.options, letters),
           "bare_question": a.question}
    out = run_record(model, processor, rec, suffix, lambda t: parse(t, letters), max_new_tokens=max_new)
    if answer is not None:
        out["answer"] = answer
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

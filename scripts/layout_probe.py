"""Run PP-DocLayoutV3 locally and retain its regions, order and overlay.

Use an environment with transformers>=5.5,<6, torch, torchvision, Pillow and
PyMuPDF. This diagnostic is independent of the production Surya environment.
It performs no recognition and does not change any OCR reference or output.
"""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="Directory for JSON and PNG evidence")
    parser.add_argument("--page", type=int, default=1, help="One-based source PDF page")
    parser.add_argument("--model", default="PaddlePaddle/PP-DocLayoutV3_safetensors")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.page < 1 or not 0 <= args.threshold <= 1:
        parser.error("page must be positive and threshold must be in [0, 1]")

    import fitz
    import torch
    import transformers
    from PIL import Image, ImageDraw
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    torch.set_num_threads(4)
    with fitz.open(args.input) as doc:
        if args.page > len(doc):
            parser.error("page is outside the source document")
        pix = doc[args.page - 1].get_pixmap(dpi=200)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    image.thumbnail((2048, 2048))
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForObjectDetection.from_pretrained(args.model).to(args.device).eval()
    started = time.monotonic()
    with torch.inference_mode():
        inputs = processor(images=image, return_tensors="pt").to(args.device)
        result = processor.post_process_object_detection(
            model(**inputs), threshold=args.threshold, target_sizes=[image.size[::-1]],
        )[0]
    elapsed = time.monotonic() - started
    regions = []
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for order, (score, label, box, polygon) in enumerate(zip(
        result["scores"], result["labels"], result["boxes"], result["polygon_points"],
    )):
        label = model.config.id2label[int(label)]
        box = box.tolist()
        regions.append({"order": order, "label": label, "score": float(score),
                        "bbox": box, "polygon": polygon.tolist()})
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{order}: {label}", fill="red",
                  stroke_width=1, stroke_fill="white")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.input.stem}-page-{args.page}"
    payload = {"source": args.input.name, "page": args.page, "model": args.model,
               "transformers": transformers.__version__, "device": args.device,
               "threshold": args.threshold, "image_size": image.size,
               "inference_seconds": elapsed, "regions": regions}
    (args.output / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    overlay.save(args.output / f"{stem}.png")
    print(f"{len(regions)} regions in {elapsed:.2f}s; evidence in {args.output}")


if __name__ == "__main__":
    main()

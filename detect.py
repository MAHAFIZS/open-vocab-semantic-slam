from pathlib import Path
import json
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Owlv2ForObjectDetection

IMAGE_PATH = Path("data/test.jpg")
OUTPUT_IMG = Path("outputs/detections.jpg")
OUTPUT_JSON = Path("outputs/detections.json")

TEXT_QUERIES = [["laptop", "bottle", "chair", "table", "book", "keyboard", "mouse"]]

def main():
    image = Image.open(IMAGE_PATH).convert("RGB")

    processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")
    model.eval()

    inputs = processor(text=TEXT_QUERIES, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=0.15,
        text_labels=TEXT_QUERIES,
    )[0]

    draw = ImageDraw.Draw(image)
    detections = []

    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        box = box.tolist()
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), f"{label}:{score:.2f}", fill="red")
        detections.append({
            "label": label,
            "score": float(score),
            "box": box,
            "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        })

    image.save(OUTPUT_IMG)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(detections, f, indent=2)

    print(f"Saved: {OUTPUT_IMG}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Detections: {len(detections)}")

if __name__ == "__main__":
    main()

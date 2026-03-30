from pathlib import Path
import json
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Owlv2ForObjectDetection

FRAMES_DIR = Path("frames")
OUT_DIR = Path("video_detections")
OUT_JSON = OUT_DIR / "all_detections.json"

TEXT_QUERIES = [["laptop", "bottle", "chair", "table", "book", "keyboard", "mouse","sofa"]]
THRESHOLD = 0.20

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")
    model.eval()

    all_results = []

    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    for idx, frame_path in enumerate(frame_paths):
        print(f"Processing {frame_path.name} ({idx+1}/{len(frame_paths)})", flush=True)
        image = Image.open(frame_path).convert("RGB")

        inputs = processor(text=TEXT_QUERIES, images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=THRESHOLD,
            text_labels=TEXT_QUERIES,
        )[0]

        draw = ImageDraw.Draw(image)
        frame_dets = []

        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 12)), f"{label}:{score:.2f}", fill="red")

            frame_dets.append({
                "label": label,
                "score": float(score),
                "box": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            })

        image.save(OUT_DIR / frame_path.name)
        all_results.append({
            "frame": frame_path.name,
            "detections": frame_dets
        })

    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved detections to {OUT_JSON}")

if __name__ == "__main__":
    main()

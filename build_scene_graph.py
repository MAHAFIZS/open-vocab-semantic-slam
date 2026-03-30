from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

DETECTIONS_PATH = Path("outputs/detections.json")
IMAGE_PATH = Path("data/test.jpg")
SCENE_GRAPH_JSON = Path("outputs/scene_graph.json")
SCENE_GRAPH_IMG = Path("outputs/scene_graph.png")

MIN_SCORE = 0.20
# Labels we are willing to keep for this demo
ALLOWED_LABELS = {"laptop", "bottle", "chair", "table", "book", "keyboard", "mouse"}


def load_detections():
    with open(DETECTIONS_PATH, "r") as f:
        detections = json.load(f)
    return detections


def filter_detections(detections):
    filtered = []
    for det in detections:
        label = det["label"]
        score = det["score"]
        if label in ALLOWED_LABELS and score >= MIN_SCORE:
            filtered.append(det)
    return filtered


def add_pseudo_3d_positions(detections, image_width, image_height):
    """
    Convert 2D box centers to a simple pseudo-3D coordinate:
    x -> normalized horizontal position
    y -> normalized vertical position
    z -> inverse box area heuristic (bigger object => closer => smaller z)
    """
    objects = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        area = w * h

        x_norm = cx / image_width
        y_norm = cy / image_height

        # Simple depth heuristic: larger box -> closer -> lower z
        z_est = 1.0 / math.sqrt(area)

        objects.append({
            "id": i,
            "label": det["label"],
            "score": det["score"],
            "box": det["box"],
            "center_2d": [cx, cy],
            "position_3d_est": [x_norm, y_norm, z_est]
        })
    return objects


def horizontal_overlap(a_box, b_box):
    ax1, _, ax2, _ = a_box
    bx1, _, bx2, _ = b_box
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    width = min(ax2 - ax1, bx2 - bx1)
    return overlap / max(1.0, width)


def infer_relations(objects, image_width, image_height):
    relations = []

    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue

            a = objects[i]
            b = objects[j]

            ax, ay, az = a["position_3d_est"]
            bx, by, bz = b["position_3d_est"]

            dx = ax - bx
            dy = ay - by
            dist = math.sqrt(dx * dx + dy * dy)

            # left_of / right_of
            if ax < bx - 0.08:
                relations.append({
                    "subject": a["label"],
                    "subject_id": a["id"],
                    "predicate": "left_of",
                    "object": b["label"],
                    "object_id": b["id"]
                })

            # above / below
            if ay < by - 0.08:
                relations.append({
                    "subject": a["label"],
                    "subject_id": a["id"],
                    "predicate": "above",
                    "object": b["label"],
                    "object_id": b["id"]
                })

            # near
            if dist < 0.20:
                relations.append({
                    "subject": a["label"],
                    "subject_id": a["id"],
                    "predicate": "near",
                    "object": b["label"],
                    "object_id": b["id"]
                })

            # on (simple heuristic)
            if (
                a["label"] != b["label"]
                and ay < by
                and abs(ax - bx) < 0.12
                and horizontal_overlap(a["box"], b["box"]) > 0.3
                and (by - ay) < 0.20
            ):
                relations.append({
                    "subject": a["label"],
                    "subject_id": a["id"],
                    "predicate": "on",
                    "object": b["label"],
                    "object_id": b["id"]
                })

    return deduplicate_relations(relations)


def deduplicate_relations(relations):
    seen = set()
    unique = []
    for r in relations:
        key = (
            r["subject"], r["subject_id"],
            r["predicate"],
            r["object"], r["object_id"]
        )
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def save_scene_graph(objects, relations):
    graph_data = {
        "objects": objects,
        "relations": relations
    }
    with open(SCENE_GRAPH_JSON, "w") as f:
        json.dump(graph_data, f, indent=2)
    return graph_data


def draw_scene_graph(objects, relations):
    G = nx.DiGraph()

    for obj in objects:
        node_name = f'{obj["label"]}_{obj["id"]}'
        G.add_node(node_name)

    for rel in relations:
        src = f'{rel["subject"]}_{rel["subject_id"]}'
        dst = f'{rel["object"]}_{rel["object_id"]}'
        G.add_edge(src, dst, label=rel["predicate"])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2500, font_size=9)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Scene Graph")
    plt.tight_layout()
    plt.savefig(SCENE_GRAPH_IMG, dpi=200)
    plt.close()


def main():
    if not DETECTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing detections file: {DETECTIONS_PATH}")

    image = Image.open(IMAGE_PATH)
    image_width, image_height = image.size

    detections = load_detections()
    filtered = filter_detections(detections)
    objects = add_pseudo_3d_positions(filtered, image_width, image_height)
    relations = infer_relations(objects, image_width, image_height)

    save_scene_graph(objects, relations)
    draw_scene_graph(objects, relations)

    print(f"Objects kept: {len(objects)}")
    print(f"Relations inferred: {len(relations)}")
    print(f"Saved: {SCENE_GRAPH_JSON}")
    print(f"Saved: {SCENE_GRAPH_IMG}")


if __name__ == "__main__":
    main()

from pathlib import Path
import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

INPUT_JSON = Path("video_detections/all_detections.json")
OUTPUT_JSON = Path("outputs/temporal_scene_graph.json")
OUTPUT_IMG = Path("outputs/temporal_scene_graph.png")

# Keep only confident detections
MIN_SCORE = 0.20

# Object must appear in at least this many unique frames
MIN_FRAMES = 5

# Larger threshold = more detections merged into one persistent object
DIST_THRESH = 200.0

# Keep at most this many persistent objects per label
MAX_PER_LABEL = 2

# Allowed labels for final graph
ALLOWED_LABELS = {
    "laptop",
    "bottle",
    "chair",
    "table",
    "book",
    "keyboard",
    "mouse",
    "sofa",
    "couch",
}

# Relation thresholds
LEFT_RIGHT_THRESH = 100.0
ABOVE_BELOW_THRESH = 100.0
NEAR_THRESH = 140.0


def load_data():
    with open(INPUT_JSON, "r") as f:
        return json.load(f)


def center_of(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def box_area(box):
    x1, y1, x2, y2 = box
    return max(1.0, (x2 - x1)) * max(1.0, (y2 - y1))


def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def collect_candidates(frames):
    detections = []
    for frame_entry in frames:
        frame_name = frame_entry["frame"]
        for det in frame_entry["detections"]:
            label = det["label"]
            score = det["score"]

            if score < MIN_SCORE:
                continue
            if label not in ALLOWED_LABELS:
                continue

            box = det["box"]
            detections.append({
                "frame": frame_name,
                "label": label,
                "score": score,
                "box": box,
                "center": det.get("center", center_of(box)),
                "area": box_area(box),
            })
    return detections


def cluster_objects(detections):
    clusters = []

    for det in detections:
        best_cluster = None
        best_dist = None

        for cluster in clusters:
            if cluster["label"] != det["label"]:
                continue

            dist = euclidean(cluster["avg_center"], det["center"])
            if dist < DIST_THRESH:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_cluster = cluster

        if best_cluster is None:
            clusters.append({
                "label": det["label"],
                "members": [det],
                "avg_center": det["center"][:],
                "avg_score": det["score"],
                "avg_area": det["area"],
            })
        else:
            best_cluster["members"].append(det)
            n = len(best_cluster["members"])

            old_cx, old_cy = best_cluster["avg_center"]
            new_cx, new_cy = det["center"]
            best_cluster["avg_center"] = [
                (old_cx * (n - 1) + new_cx) / n,
                (old_cy * (n - 1) + new_cy) / n,
            ]
            best_cluster["avg_score"] = (
                (best_cluster["avg_score"] * (n - 1) + det["score"]) / n
            )
            best_cluster["avg_area"] = (
                (best_cluster["avg_area"] * (n - 1) + det["area"]) / n
            )

    return clusters


def finalize_objects(clusters):
    persistent = []
    obj_id = 0

    for cluster in clusters:
        unique_frames = sorted(set(m["frame"] for m in cluster["members"]))
        if len(unique_frames) < MIN_FRAMES:
            continue

        xs = [m["center"][0] for m in cluster["members"]]
        ys = [m["center"][1] for m in cluster["members"]]

        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)

        persistent.append({
            "id": obj_id,
            "label": cluster["label"],
            "frames_seen": len(unique_frames),
            "frame_names": unique_frames,
            "avg_score": cluster["avg_score"],
            "avg_area": cluster["avg_area"],
            "center_2d": [avg_x, avg_y],
        })
        obj_id += 1

    return persistent


def limit_per_label(objects, max_per_label=MAX_PER_LABEL):
    grouped = defaultdict(list)
    for obj in objects:
        grouped[obj["label"]].append(obj)

    final = []
    new_id = 0

    for label, objs in grouped.items():
        objs = sorted(
            objs,
            key=lambda x: (-x["frames_seen"], -x["avg_score"], -x["avg_area"])
        )
        keep = objs[:max_per_label]
        for obj in keep:
            obj = dict(obj)
            obj["id"] = new_id
            final.append(obj)
            new_id += 1

    return final


def infer_relations(objects):
    relations = []

    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue

            a = objects[i]
            b = objects[j]

            ax, ay = a["center_2d"]
            bx, by = b["center_2d"]

            dx = ax - bx
            dy = ay - by
            dist = math.sqrt(dx * dx + dy * dy)

            if ax < bx - LEFT_RIGHT_THRESH:
                relations.append({
                    "subject": a["label"],
                    "subject_id": a["id"],
                    "predicate": "left_of",
                    "object": b["label"],
                    "object_id": b["id"],
                    "distance": dist,
                })

            if ay < by - ABOVE_BELOW_THRESH:
                relations.append({
                    "subject": a["label"],
                    "subject_id": a["id"],
                    "predicate": "above",
                    "object": b["label"],
                    "object_id": b["id"],
                    "distance": dist,
                })

            if dist < NEAR_THRESH:
                relations.append({
                    "subject": a["label"],
                    "subject_id": a["id"],
                    "predicate": "near",
                    "object": b["label"],
                    "object_id": b["id"],
                    "distance": dist,
                })

    return reduce_relations(relations)


def reduce_relations(relations):
    grouped = defaultdict(list)

    for r in relations:
        key = (r["subject"], r["subject_id"], r["predicate"])
        grouped[key].append(r)

    reduced = []
    seen = set()

    for _, rels in grouped.items():
        rels = sorted(rels, key=lambda x: x["distance"])
        for r in rels[:2]:
            dedup_key = (
                r["subject"], r["subject_id"],
                r["predicate"],
                r["object"], r["object_id"]
            )
            if dedup_key not in seen:
                seen.add(dedup_key)
                r = dict(r)
                r.pop("distance", None)
                reduced.append(r)

    return reduced


def save_graph(objects, relations):
    data = {
        "objects": objects,
        "relations": relations
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    return data


def draw_graph(objects, relations):
    G = nx.DiGraph()

    for obj in objects:
        name = f'{obj["label"]}_{obj["id"]}'
        G.add_node(name)

    for rel in relations:
        src = f'{rel["subject"]}_{rel["subject_id"]}'
        dst = f'{rel["object"]}_{rel["object_id"]}'
        G.add_edge(src, dst, label=rel["predicate"])

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=1.2)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2800,
        font_size=10,
        arrows=True
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8
    )

    plt.title("Temporal Scene Graph")
    plt.savefig(OUTPUT_IMG, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_JSON}")

    frames = load_data()
    detections = collect_candidates(frames)
    clusters = cluster_objects(detections)
    objects = finalize_objects(clusters)
    objects = limit_per_label(objects, max_per_label=MAX_PER_LABEL)
    relations = infer_relations(objects)

    save_graph(objects, relations)
    draw_graph(objects, relations)

    print(f"Raw detections considered: {len(detections)}")
    print(f"Clusters found: {len(clusters)}")
    print(f"Persistent objects kept: {len(objects)}")
    print(f"Relations inferred: {len(relations)}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_IMG}")


if __name__ == "__main__":
    main()
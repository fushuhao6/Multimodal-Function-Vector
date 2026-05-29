"""
prepare_gqa_dataset.py

Generic GQA dataset constructor for the Multimodal-Function-Vector repo.
Filters GQA scene graphs by configurable relations, thresholds, and object rules.

Example usage:
    python dataset/prepare_gqa_dataset.py \
        --relations "wearing" "holding" "carrying" "sitting on" "standing on" "standing in" "riding" \
        --output_name gqa_agentic

    python dataset/prepare_gqa_dataset.py \
        --relations "above" "below" "behind" "next to" \
        --output_name gqa_spatial

    python dataset/prepare_gqa_dataset.py \
        --relations "holding" "riding" \
        --min_objects 2 --max_objects 8 --min_relations 2 \
        --max_duplicate_objects 2 \
        --output_name pilot_dataset
"""

import json
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RAW_DIR       = REPO_ROOT / "data" / "gqa" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "gqa" / "processed"

SCENE_FILE_NAMES = ["train_sceneGraphs.json", "val_sceneGraphs.json"]

DEFAULT_NOT_ALLOWED_OBJECTS = {
    "sky", "wall", "floor", "ceiling", "building", "trees", "grass", "ocean",
    "water", "fence", "bush", "neck", "hair", "legs", "rocks", "bushes",
    "clouds", "cloud", "sun", "stars", "cars", "leaves", "fingers", "hand", "arm",
}

MAX_RELATION_TOKENS_IN_FILENAME = 3  # truncate after this many relation names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter GQA scene graphs into a dataset JSON for Multimodal-Function-Vector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required ---
    parser.add_argument(
        "--relations",
        nargs="+",
        required=True,
        metavar="RELATION",
        help='One or more relation predicates to keep. Supports multi-word values, e.g. "sitting on".',
    )

    # --- paths ---
    parser.add_argument(
        "--raw_data_dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing train_sceneGraphs.json and val_sceneGraphs.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory where the filtered JSON will be saved.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help=(
            "Output filename stem (without .json). "
            "Auto-generated from relation names if omitted."
        ),
    )

    # --- object size thresholds ---
    parser.add_argument("--min_obj_ratio", type=float, default=0.05,
                        help="Minimum object area as a fraction of image area.")
    parser.add_argument("--max_obj_ratio", type=float, default=0.5,
                        help="Maximum object area as a fraction of image area.")

    # --- object count thresholds ---
    parser.add_argument("--min_objects", type=int, default=3,
                        help="Minimum number of valid objects per image.")
    parser.add_argument("--max_objects", type=int, default=30,
                        help="Maximum number of valid objects per image.")

    # --- relation count thresholds ---
    parser.add_argument("--min_relations", type=int, default=1,
                        help="Minimum number of matching relations per image.")
    parser.add_argument("--max_relations", type=int, default=15,
                        help="Maximum number of matching relations per image.")
    parser.add_argument("--min_different_relations", type=int, default=1,
                        help="Minimum number of distinct relation predicates per image.")

    # --- duplicate object policy ---
    parser.add_argument(
        "--max_duplicate_objects",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Maximum allowed occurrences of the same object name. "
            "1 = unique objects only; 2 = allow up to two of the same name; etc."
        ),
    )

    # --- predicate ambiguity policy ---
    parser.add_argument(
        "--allow_multi_object_predicates",
        action="store_true",
        default=False,
        help=(
            "If set, allow the same subject to use the same predicate with multiple objects "
            '(e.g. "person holding phone" AND "person holding book" in the same image). '
            "By default such images are excluded."
        ),
    )

    # --- object blacklist ---
    parser.add_argument(
        "--not_allowed_objects",
        nargs="+",
        default=DEFAULT_NOT_ALLOWED_OBJECTS,
        metavar="OBJECT",
        help=(
            "Object names to exclude. Defaults to the built-in blacklist. "
            "Names are lowercased automatically."
        ),
    )

    return parser.parse_args()

def validate_args(args):
    if args.min_objects > args.max_objects:
        raise ValueError(
            "min_objects cannot exceed max_objects"
        )

    if args.min_relations > args.max_relations:
        raise ValueError(
            "min_relations cannot exceed max_relations"
        )

    if args.min_obj_ratio > args.max_obj_ratio:
        raise ValueError(
            "min_obj_ratio cannot exceed max_obj_ratio"
        )

    if args.max_duplicate_objects < 1:
        raise ValueError(
            "max_duplicate_objects must be >= 1"
        )

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_scene_graphs(raw_data_dir: Path) -> dict:
    """Load and merge train + val scene graph files."""
    merged: dict = {}
    for fname in SCENE_FILE_NAMES:
        fpath = raw_data_dir / fname
        print(f"Loading {fpath} ...")
        if not fpath.exists():
            raise FileNotFoundError(f"Expected file not found: {fpath}")
        with open(fpath, "r") as f:
            data = json.load(f)
        print(f"  Loaded {len(data):,} images from {fname}")
        merged.update(data)
    print(f"Total images loaded: {len(merged):,}\n")
    return merged


def build_output_path(output_dir: Path, output_name: Optional[str], relations: List[str]) -> Path:
    """Determine the full output file path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name:
        stem = output_name
    else:
        tokens = [r.replace(" ", "_") for r in relations]
        if len(tokens) <= MAX_RELATION_TOKENS_IN_FILENAME:
            stem = "gqa_" + "_".join(tokens)
        else:
            stem = "gqa_" + "_".join(tokens[:MAX_RELATION_TOKENS_IN_FILENAME]) + "_etc"

    return output_dir / f"{stem}.json"


# ---------------------------------------------------------------------------
# Per-image helpers
# ---------------------------------------------------------------------------

def filter_valid_objects(
    objects: dict,
    image_w: int,
    image_h: int,
    min_ratio: float,
    max_ratio: float,
    blacklist: set,
) -> dict:
    """Return the subset of objects that pass size and blacklist filters."""
    img_area = image_w * image_h
    if img_area == 0:
        return {}

    valid: dict = {}
    for obj_id, obj in objects.items():
        x, y, w, h = obj.get("x", 0), obj.get("y", 0), obj.get("w", 0), obj.get("h", 0)
        ratio = (w * h) / img_area
        if not (min_ratio <= ratio <= max_ratio):
            continue
        name = obj.get("name", "").lower()
        if name in blacklist:
            continue
        valid[obj_id] = {"name": name, "bbox": {"x": x, "y": y, "w": w, "h": h}}

    return valid


def passes_duplicate_constraint(valid_objects: dict, max_duplicates: int) -> bool:
    """Return True if no object name appears more than max_duplicates times."""
    counts = Counter(obj["name"] for obj in valid_objects.values())
    return all(c <= max_duplicates for c in counts.values())


def extract_relations(objects: dict, valid_objects: dict, allowed_relations: set) -> list:
    """Extract relations where both subject and object are in valid_objects."""
    relations: list = []
    for subj_id, subj in objects.items():
        if subj_id not in valid_objects:
            continue
        for rel in subj.get("relations", []):
            obj_id = rel.get("object")
            if obj_id not in valid_objects:
                continue
            predicate = rel.get("name", "").lower()
            if predicate in allowed_relations:
                relations.append({
                    "subject": valid_objects[subj_id]["name"],
                    "object": valid_objects[obj_id]["name"],
                    "predicate": predicate,
                })
    return relations


def passes_relation_constraints(
    relations: list,
    min_relations: int,
    max_relations: int,
    min_different: int,
    allow_multi_object_predicates: bool,
) -> bool:
    """Return True if the relation list satisfies all count and ambiguity rules."""
    n = len(relations)
    if not (min_relations <= n <= max_relations):
        return False

    unique_predicates = {r["predicate"] for r in relations}
    if len(unique_predicates) < min_different:
        return False

    if not allow_multi_object_predicates:
        subj_pred_count = Counter((r["subject"], r["predicate"]) for r in relations)
        if any(c > 1 for c in subj_pred_count.values()):
            return False

    return True


# ---------------------------------------------------------------------------
# Main filtering loop
# ---------------------------------------------------------------------------

def process_scene_graphs(scene_graphs: dict, args: argparse.Namespace, blacklist: set) -> List[Dict]:
    """Apply all filtering rules and return the list of passing images."""
    allowed_relations = {r.lower() for r in args.relations}

    items = scene_graphs.items()
    if TQDM_AVAILABLE:
        items = tqdm(items, total=len(scene_graphs), desc="Filtering images", unit="img")

    results: list = []

    for img_id, graph in items:
        raw_objects = graph.get("objects", {})
        if not raw_objects:
            continue

        image_w = graph.get("width", 0)
        image_h = graph.get("height", 0)
        if not image_w or not image_h:
            continue

        # Step 1: size + blacklist filter
        valid_objects = filter_valid_objects(
            raw_objects, image_w, image_h,
            args.min_obj_ratio, args.max_obj_ratio, blacklist,
        )

        # Step 2: object count check
        if not (args.min_objects <= len(valid_objects) <= args.max_objects):
            continue

        # Step 3: duplicate object check
        if not passes_duplicate_constraint(valid_objects, args.max_duplicate_objects):
            continue

        # Step 4: extract matching relations
        relations = extract_relations(raw_objects, valid_objects, allowed_relations)

        # Step 5: relation constraint check
        if not passes_relation_constraints(
            relations,
            args.min_relations,
            args.max_relations,
            args.min_different_relations,
            args.allow_multi_object_predicates,
        ):
            continue

        results.append({
            "image_id": img_id,
            "objects": list(valid_objects.values()),
            "relations": relations,
        })

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate_args(args)

    # Resolve blacklist
    if args.not_allowed_objects is not None:
        blacklist = {name.lower() for name in args.not_allowed_objects}
    else:
        blacklist = DEFAULT_NOT_ALLOWED_OBJECTS

    # Print configuration summary
    print("=" * 60)
    print("GQA Dataset Preparation")
    print("=" * 60)
    print(f"  Relations         : {sorted(args.relations)}")
    print(f"  Object ratio      : [{args.min_obj_ratio}, {args.max_obj_ratio}]")
    print(f"  Object count      : [{args.min_objects}, {args.max_objects}]")
    print(f"  Relation count    : [{args.min_relations}, {args.max_relations}]")
    print(f"  Min unique rels   : {args.min_different_relations}")
    print(f"  Max dup objects   : {args.max_duplicate_objects}")
    print(f"  Multi-obj preds   : {'allowed' if args.allow_multi_object_predicates else 'excluded'}")
    print(f"  Blacklist size    : {len(blacklist)} objects")
    print()

    # Load data
    scene_graphs = load_scene_graphs(args.raw_data_dir)

    # Filter
    results = process_scene_graphs(scene_graphs, args, blacklist)

    # Save
    output_path = build_output_path(args.output_dir, args.output_name, args.relations)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results):,} filtered images to:\n  {output_path}")


if __name__ == "__main__":
    main()
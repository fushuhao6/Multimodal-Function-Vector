"""
split_gqa_dataset.py

Generic train/test splitter for any GQA-format filtered dataset JSON.
Designed to work with output from prepare_gqa_dataset.py, but accepts
any JSON matching the schema: [{image_id, objects, relations}, ...]

Example usage:
    python dataset/split_gqa_dataset.py \
        --input data/gqa/processed/gqa_agentic.json

    python dataset/split_gqa_dataset.py \
        --input data/gqa/processed/gqa_spatial.json \
        --split_strategy random \
        --train_out data/gqa/processed/spatial_train.json \
        --test_out  data/gqa/processed/spatial_test.json

    python dataset/split_gqa_dataset.py \
        --input data/gqa/processed/gqa_agentic.json \
               data/gqa/processed/gqa_spatial.json \
        --merge_duplicates 
"""

import json
import random
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a GQA-format filtered JSON dataset into train and test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more paths to filtered GQA JSON files (list of {image_id, objects, relations}).",
    )
    parser.add_argument(
        "--train_out",
        type=Path,
        default=None,
        help=(
            "Output path for the train split. "
            "Defaults to <input_stem>_train.json in the same directory as the (first) input file."
        ),
    )
    parser.add_argument(
        "--test_out",
        type=Path,
        default=None,
        help=(
            "Output path for the test split. "
            "Defaults to <input_stem>_test.json in the same directory as the (first) input file."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--split_strategy",
        choices=["greedy", "random"],
        default="greedy",
        help=(
            "Split strategy. "
            "'greedy' balances relation-class distributions between splits; "
            "'random' shuffles and splits 50/50."
        ),
    )
    parser.add_argument(
        "--merge_duplicates",
        action="store_true",
        default=False,
        help=(
            "If set, merge items that share the same image_id by combining their "
            "objects and relations before splitting. Off by default."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O and validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"image_id", "objects", "relations"}


def load_and_validate(path: Path) -> list:
    """Load a JSON file and validate that every item has the expected schema."""
    print(f"Loading {path} ...")
    data = json.loads(path.read_text())

    if isinstance(data, dict):
        items = []
        for k, v in data.items():
            entry = dict(v)
            entry.setdefault("image_id", k)
            items.append(entry)
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"{path}: unsupported JSON format — must be a list or dict.")

    # Schema validation
    errors: list[str] = []
    for i, item in enumerate(items):
        missing = REQUIRED_KEYS - item.keys()
        if missing:
            errors.append(f"  item[{i}] missing keys: {sorted(missing)}")
        if errors and i >= 10:
            errors.append("  ... (too many errors, stopping early)")
            break

    if errors:
        print(f"ERROR: schema validation failed for {path}:")
        for e in errors:
            print(e)
        sys.exit(1)

    print(f"  Loaded {len(items):,} items.")
    return items


def merge_by_image_id(items: list) -> list:
    """Merge items that share an image_id by combining objects and relations."""
    merged: dict = {}
    for item in items:
        img_id = item["image_id"]
        if img_id not in merged:
            merged[img_id] = {
                "image_id": img_id,
                "objects": list(item.get("objects", [])),
                "relations": list(item.get("relations", [])),
            }
            # carry over any extra keys from the first occurrence
            for k, v in item.items():
                if k not in merged[img_id]:
                    merged[img_id][k] = v
        else:
            merged[img_id]["objects"].extend(item.get("objects", []))
            merged[img_id]["relations"].extend(item.get("relations", []))
    return list(merged.values())


def build_output_paths(
    input_paths: list[Path],
    train_out: Optional[Path],
    test_out: Optional[Path],
) -> tuple[Path, Path]:
    """Derive output paths from the first input path if not explicitly set."""
    primary = input_paths[0].resolve()
    stem = primary.stem          # e.g. "gqa_agentic"
    out_dir = primary.parent

    resolved_train = train_out if train_out else out_dir / f"{stem}_train.json"
    resolved_test  = test_out  if test_out  else out_dir / f"{stem}_test.json"
    return resolved_train, resolved_test


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------

def _per_image_rel_counter(item: dict) -> Counter:
    return Counter(r["predicate"] for r in item.get("relations", []))


def random_split(items: list, seed: int) -> tuple[list, list]:
    """Shuffle and split 50/50 (train gets the extra item if odd total)."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    mid = (len(shuffled) + 1) // 2
    return shuffled[:mid], shuffled[mid:]


def greedy_balanced_split(items: list, seed: int) -> tuple[list, list]:
    """
    Greedy split that minimises the absolute difference in per-predicate
    relation counts between the two halves.
    """
    rng = random.Random(seed)
    img_counters = [_per_image_rel_counter(it) for it in items]

    global_counts: Counter = Counter()
    for c in img_counters:
        global_counts.update(c)
    target = Counter({p: v / 2.0 for p, v in global_counts.items()})

    # Process heavier images first (more predictable greedy choices)
    order = sorted(range(len(items)), key=lambda i: -sum(img_counters[i].values()))
    rng.shuffle(order)  # break ties randomly but reproducibly

    n = len(items)
    train_cap = (n + 1) // 2
    test_cap  = n // 2

    train_idx, test_idx = [], []
    train_counts, test_counts = Counter(), Counter()

    def _imbalance(current: Counter, added: Counter) -> float:
        keys = set(current) | set(added) | set(target)
        return sum(abs(current[p] + added[p] - target[p]) for p in keys)

    for i in order:
        ic = img_counters[i]

        if len(train_idx) >= train_cap:
            test_idx.append(i); test_counts.update(ic); continue
        if len(test_idx) >= test_cap:
            train_idx.append(i); train_counts.update(ic); continue

        train_score = _imbalance(train_counts, ic)
        test_score  = _imbalance(test_counts,  ic)

        if train_score < test_score:
            train_idx.append(i); train_counts.update(ic)
        elif test_score < train_score:
            test_idx.append(i);  test_counts.update(ic)
        else:
            # tie: balance by image count
            if len(train_idx) <= len(test_idx):
                train_idx.append(i); train_counts.update(ic)
            else:
                test_idx.append(i);  test_counts.update(ic)

    return [items[i] for i in train_idx], [items[i] for i in test_idx]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(name: str, items: list) -> tuple[Counter, int]:
    rels: Counter = Counter()
    for it in items:
        rels.update(r["predicate"] for r in it.get("relations", []))
    total = sum(rels.values())
    print(f"\n== {name} ==")
    print(f"Images: {len(items):,}  |  Relation instances: {total:,}")
    if not total:
        print("  (no relations)")
        return rels, total
    for k, v in sorted(rels.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:22s}  {v:6,d}  ({v / total * 100:6.2f}%)")
    return rels, total


def print_distribution_diff(rel_train: Counter, tot_train: int,
                             rel_test: Counter,  tot_test:  int) -> None:
    if not tot_train or not tot_test:
        return
    all_preds = set(rel_train) | set(rel_test)
    print("\n== Distribution difference (Train% − Test%) ==")
    for p in sorted(all_preds):
        tp = rel_train[p] / tot_train * 100.0
        ep = rel_test[p]  / tot_test  * 100.0
        print(f"  {p:22s}  {tp - ep:+6.2f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Load ---------------------------------------------------------------
    print("=" * 60)
    print("GQA Dataset Splitter")
    print("=" * 60)
    print(f"  Strategy  : {args.split_strategy}")
    print(f"  Seed      : {args.seed}")
    print(f"  Merge dups: {args.merge_duplicates}")
    print()

    input_paths = [Path(p) for p in args.input]
    for p in input_paths:
        if not p.exists():
            print(f"ERROR: input file not found: {p}")
            sys.exit(1)

    all_items: list = []
    unique_per_file: dict[str, int] = {}
    for p in input_paths:
        loaded = load_and_validate(p)
        unique_per_file[p.name] = len({it["image_id"] for it in loaded})
        all_items.extend(loaded)

    print("\n=== Unique image counts per input file ===")
    for fname, count in unique_per_file.items():
        print(f"  {fname}: {count:,} unique images")

    # ---- Merge --------------------------------------------------------------
    if args.merge_duplicates:
        before = len(all_items)
        all_items = merge_by_image_id(all_items)
        print(f"\nMerged {before:,} items → {len(all_items):,} unique images.")
    else:
        # Still deduplicate silently if multiple files contributed the same image_id
        seen_ids = set()
        deduped: list = []
        for it in all_items:
            if it["image_id"] not in seen_ids:
                seen_ids.add(it["image_id"])
                deduped.append(it)
        if len(deduped) < len(all_items):
            print(
                f"\nNote: dropped {len(all_items) - len(deduped):,} duplicate image_id entries "
                f"(use --merge_duplicates to combine them instead)."
            )
        all_items = deduped

    print(f"\nTotal images going into split: {len(all_items):,}")

    # ---- Split --------------------------------------------------------------
    if args.split_strategy == "greedy":
        train, test = greedy_balanced_split(all_items, seed=args.seed)
    else:
        train, test = random_split(all_items, seed=args.seed)

    train_ids = {it["image_id"] for it in train}
    test_ids  = {it["image_id"] for it in test}
    overlap   = train_ids & test_ids

    print(f"\nTrain images : {len(train):,}")
    print(f"Test images  : {len(test):,}")
    print(f"Overlap      : {len(overlap)} (should be 0)")
    if overlap:
        print(f"WARNING: overlapping image_ids detected: {list(overlap)[:5]} ...")

    # ---- Metadata -----------------------------------------------------------
    metadata = {
        "seed": args.seed,
        "split_strategy": args.split_strategy,
        "train_size": len(train),
        "test_size": len(test),
        "input_files": [str(p) for p in input_paths],
    }

    # ---- Save ---------------------------------------------------------------
    train_path, test_path = build_output_paths(input_paths, args.train_out, args.test_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_path.write_text(json.dumps(train, indent=2))
    test_path.write_text(json.dumps(test, indent=2))

    meta_path = train_path.parent / f"{train_path.stem.removesuffix('_train')}_split_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved train split  → {train_path}")
    print(f"Saved test split   → {test_path}")
    print(f"Saved metadata     → {meta_path}")

    # ---- Stats --------------------------------------------------------------
    rel_train, tot_train = summarize("TRAIN", train)
    rel_test,  tot_test  = summarize("TEST",  test)
    print_distribution_diff(rel_train, tot_train, rel_test, tot_test)


if __name__ == "__main__":
    main()
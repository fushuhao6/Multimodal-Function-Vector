from __future__ import annotations
from pathlib import Path
import json
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from collections import Counter

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from dataset.utils import construct_question

PathLike = Union[str, Path]

def _xywh_to_xyxy(x, y, w, h):
    return [x, y, x + w, y + h]


def find_duplicate_object_names(annotations):
    """
    Check for duplicate object names in each image annotation.

    Args:
        annotations: dict mapping image_id -> {"objects": [...], "relations": [...]}
                     or list of such entries with "image_id" key.

    Returns:
        dict mapping image_id -> list of duplicate names (empty list if none)
    """
    results = {}

    # Normalize to dict form
    if isinstance(annotations, list):
        ann_by_id = {str(e["image_id"]): e for e in annotations}
    elif isinstance(annotations, dict):
        ann_by_id = annotations
    else:
        raise TypeError("Unsupported annotations type: must be dict or list.")

    for image_id, entry in ann_by_id.items():
        names = [obj["name"] for obj in entry.get("objects", [])]
        counts = Counter(names)
        dups = [n for n, c in counts.items() if c > 1]
        results[image_id] = dups

    return results


class GQADataset(Dataset):
    """
    JSON supported shapes:
      1) List[ { "image_id": str, "objects": [...], "relations": [...] }, ... ]
      2) Dict[ image_id -> { "objects": [...], "relations": [...] } ]

    Directory layout:
      img_dir/
        <image_id>.jpg   (by default; customize via image_resolver or image_ext)
      ann_json: gqa_filtered_spatial.json

    __getitem__ returns:
      img (Tensor if transform provided, else PIL.Image)
      target: {
        "boxes": FloatTensor [N,4]  (x1,y1,x2,y2),
        "labels": LongTensor [N],
        "obj_names": List[str],
        "rel_triplets": LongTensor [R,3]  (sub_idx, pred_idx, obj_idx),
        "image_id": str,
      }
    """
    def __init__(
        self,
        img_dir: PathLike,
        ann_json: PathLike,
        transform: Optional[Callable] = None,
        image_ext: str = ".jpg",
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.image_ext = image_ext

        # ---- Load all annotations ----
        with open(ann_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        dups = find_duplicate_object_names(raw)

        # Print only the images that actually have duplicates
        for img_id, dup_names in dups.items():
            if dup_names:
                print(img_id, dup_names)

        # Normalize to a dict: image_id -> {"objects": [...], "relations": [...]}
        if isinstance(raw, list):
            self.ann_by_id = {str(e["image_id"]): {"objects": e.get("objects", []),
                                                   "relations": e.get("relations", [])}
                              for e in raw}
        elif isinstance(raw, dict):
            # if already keyed by image_id, ensure consistent fields
            self.ann_by_id = {}
            for k, v in raw.items():
                self.ann_by_id[str(k)] = {
                    "objects": v.get("objects", []),
                    "relations": v.get("relations", []),
                }
        else:
            raise TypeError("Unsupported JSON structure: must be list or dict.")

        self.image_ids = sorted(self.ann_by_id.keys())

        # ---- Build vocabularies ----
        obj_names_set, pred_set = set(), set()
        for entry in self.ann_by_id.values():
            for o in entry["objects"]:
                obj_names_set.add(o["name"])
            for r in entry["relations"]:
                pred_set.add(r["predicate"])
        self.class_to_idx = {c: i for i, c in enumerate(sorted(obj_names_set))}
        self.idx_to_class = [None] * len(self.class_to_idx)
        for k, v in self.class_to_idx.items():
            self.idx_to_class[v] = k

        self.pred_to_idx =  {p: i for i, p in enumerate(sorted(pred_set))}
        self.idx_to_pred = [None] * len(self.pred_to_idx)
        for k, v in self.pred_to_idx.items():
            self.idx_to_pred[v] = k

    def __len__(self) -> int:
        return len(self.image_ids)

    def _resolve_image_path(self, image_id: str) -> Path:
        return self.img_dir / f"{image_id}{self.image_ext}"

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        entry = self.ann_by_id[image_id]

        img = Image.open(self._resolve_image_path(image_id)).convert("RGB")

        # Objects
        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        obj_names: List[str] = []
        name_to_idx: Dict[str, int] = {}

        for i, o in enumerate(entry["objects"]):
            nm = o["name"]
            b = o["bbox"]
            boxes_xyxy.append(_xywh_to_xyxy(b["x"], b["y"], b["w"], b["h"]))
            labels.append(self.class_to_idx[nm])  # KeyError if unseen
            obj_names.append(nm)
            if nm not in name_to_idx:  # if duplicates, keep first occurrence
                name_to_idx[nm] = i

        boxes = torch.tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy else torch.zeros((0, 4),
                                                                                             dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        # Relations
        triplets: List[List[int]] = []
        for r in entry["relations"]:
            s_name, p_name, o_name = r["subject"], r["predicate"], r["object"]
            if s_name in name_to_idx and o_name in name_to_idx and p_name in self.pred_to_idx:
                triplets.append([name_to_idx[s_name], self.pred_to_idx[p_name], name_to_idx[o_name]])
            # else: skip silently; add prints if you want debugging

        rel_triplets = torch.tensor(triplets, dtype=torch.long) if triplets else torch.zeros((0, 3), dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        target = {
            "boxes": boxes,
            "labels": labels_t,
            "obj_names": obj_names,
            "rel_triplets": rel_triplets,
            "image_id": image_id,
        }
        return img, target


class ICLRelationDataset(Dataset):
    """
    Build in-context learning tasks for VLMs from a GQA-style annotations JSON.

    Annotations JSON (single file) supports:
      1) List[ { "image_id": str, "objects": [...], "relations": [...] }, ... ]
      2) Dict[ image_id -> { "objects": [...], "relations": [...] } ]

    Directory:
      img_dir/
        <image_id>.jpg  (customize via image_ext or image_resolver)

    Each item (task) returns:
      images: List[Tensor or PIL]  # length = num_context + 1
      output: str  # "s1:o1, s2:o2, ..., sK:oK, s_{K+1}:"
    where K = num_context.

    Args:
      img_dir: folder containing images.
      ann_json: path to single JSON file with all annotations.
      num_context: number of in-context examples per task (K).
      transform: optional torchvision-style transform applied to each image.
      image_ext: default ".jpg" if using <image_id>.jpg naming.
      image_resolver: optional callable image_id -> filename (e.g., lambda x: f"{x}.png")
      seed: random seed for reproducible sampling.
      allow_reuse_images: if True, images may repeat across tasks; if False, we try not to
                          reuse within the same relation, falling back if needed.
    """

    def __init__(
            self,
            data_root: PathLike,
            model_config: Dict[str, Any],
            json_file: PathLike = None,
            num_context: int = 4,
            image_ext: str = ".jpg",
            seed: int = 1234,
            use_shuffled: bool = False,
            tasks_per_relation: int = 200,
            output_zero_shot: bool = False,
    ):
        self.img_dir = Path(os.path.join(data_root, 'images'))
        self.ann_json = Path(os.path.join(data_root, 'gqa_filtered_spatial_train.json')) if json_file is None else json_file
        self.model_config = model_config
        self.image_ext = image_ext
        self.num_context = int(num_context)
        self.rng = np.random.default_rng(seed)
        self.use_shuffled = bool(use_shuffled)
        self.tasks_per_relation = int(tasks_per_relation)
        self.output_zero_shot = bool(output_zero_shot)

        # ---- Load and normalize annotations ----
        with open(self.ann_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            self.ann_by_id: Dict[str, Dict] = {
                str(e["image_id"]): {
                    "objects": e.get("objects", []),
                    "relations": e.get("relations", []),
                }
                for e in raw
            }
        elif isinstance(raw, dict):
            self.ann_by_id = {
                str(k): {
                    "objects": v.get("objects", []),
                    "relations": v.get("relations", []),
                }
                for k, v in raw.items()
            }
        else:
            raise TypeError("Unsupported JSON structure: must be list or dict.")

        self.image_ids = sorted(self.ann_by_id.keys())

        # ---- Build a fast lookup: per-image relation -> list[(subject, object)] ----
        # Example: per_image_rel_pairs[img_id]["behind"] = [("window","couch"), ...]
        self.per_image_rel_pairs: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        for img_id, entry in self.ann_by_id.items():
            rel_map: Dict[str, List[Tuple[str, str]]] = {}
            for r in entry.get("relations", []):
                p = r["predicate"]
                s = r["subject"]
                o = r["object"]
                rel_map.setdefault(p, []).append((s, o))
            self.per_image_rel_pairs[img_id] = rel_map

        # ---- Mapping: relation -> list of image_ids that contain at least one such relation ----
        self.relation_to_image_ids: Dict[str, List[str]] = {}
        for img_id, rel_map in self.per_image_rel_pairs.items():
            for predicate in rel_map.keys():
                self.relation_to_image_ids.setdefault(predicate, []).append(img_id)

        # ---- Precompute ICL tasks ----
        # Each task: {
        #   "relation": str,
        #   "image_ids": [ctx_ids..., test_id],
        #   "pairs": [(s,o), ...],
        #   "shuffled_image_ids": [ctx_ids..., test],
        #   "shuffled_pairs": [(s,o), ...],
        #   "test_image": <basename>,
        # }
        needed = self.num_context + 1
        self.tasks: Dict[str, List[Dict[str, Any]]] = {}

        # cache all relations that exist
        all_relations = list(self.relation_to_image_ids.keys())
        task_id = 0

        for relation, ids in self.relation_to_image_ids.items():
            self.tasks[relation] = []
            if len(ids) < needed:
                print(f"Relation {relation} does not have enough images "
                      f"({len(ids)} < {needed}); skipping.")
                continue

            # Try to build `tasks_per_relation` tasks; allow image reuse across tasks.
            # We allow some failures (e.g., rare missing pairs) without aborting the loop.
            attempts = 0
            max_attempts = max(10 * self.tasks_per_relation, 100)
            while len(self.tasks[relation]) < self.tasks_per_relation and attempts < max_attempts:
                task = self._build_single_task_for_relation(relation, ids, all_relations, task_id)
                if task is not None:
                    self.tasks[relation].append(task)
                    task_id += 1
                attempts += 1

            if not self.tasks[relation]:
                print(f"Failed to build any task for relation '{relation}'.")
            else:
                # Shuffle the tasks within the relation for variety
                self.rng.shuffle(self.tasks[relation])
        print(f"==================== Loaded {self.__len__()} tasks for GQA dataset =====================")

    def _build_single_task_for_relation(
            self,
            relation: str,
            ids: List[str],
            all_relations: List[str],
            task_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Make one task for `relation` using distinct images within the task.
        Returns the task dict or None if it failed to assemble a valid task.
        """
        needed = self.num_context + 1
        if len(ids) < needed:
            return None

        # --- choose K+1 distinct images, randomly assign one as test ---
        chosen = self.rng.choice(ids, size=needed, replace=False).tolist()
        test_pos = int(self.rng.integers(needed))
        test_img_id = chosen[test_pos]
        ctx_ids = [cid for i, cid in enumerate(chosen) if i != test_pos]
        chosen_ids = ctx_ids + [test_img_id]  # contexts first, test last

        # --- pick one (s,o) pair for each chosen image for THIS relation ---
        pairs: List[Tuple[str, str]] = []
        for img_id in chosen_ids:
            rel_pairs = self.per_image_rel_pairs[img_id].get(relation, [])
            if not rel_pairs:
                return None
            pairs.append(rel_pairs[self.rng.integers(len(rel_pairs))])

        test_pair = pairs[-1]

        # --- shuffled condition: round-robin over relations, unique images within task ---
        rels_perm = self.rng.permutation(all_relations).tolist()
        used_ids = set(chosen_ids)  # avoid any duplication within task
        shuf_ctx_ids: List[str] = []
        shuf_ctx_pairs: List[Tuple[str, str]] = []

        rel_ptr = 0
        attempts = 0
        max_attempts = 10 * self.num_context * max(1, len(rels_perm))
        while len(shuf_ctx_ids) < self.num_context and attempts < max_attempts:
            r = rels_perm[rel_ptr % len(rels_perm)]
            cand = [cid for cid in self.relation_to_image_ids.get(r, []) if cid not in used_ids]
            if cand:
                cid = cand[self.rng.integers(len(cand))]
                rel_pairs_r = self.per_image_rel_pairs[cid].get(r, [])
                if rel_pairs_r:
                    pair_r = rel_pairs_r[self.rng.integers(len(rel_pairs_r))]
                    shuf_ctx_ids.append(cid)
                    shuf_ctx_pairs.append(pair_r)
                    used_ids.add(cid)
            rel_ptr += 1
            attempts += 1

        # Fallback fill if still short
        if len(shuf_ctx_ids) < self.num_context:
            deficit = self.num_context - len(shuf_ctx_ids)
            pool: List[str] = []
            for rr in all_relations:
                for cid in self.relation_to_image_ids.get(rr, []):
                    if cid not in used_ids and self.per_image_rel_pairs[cid]:
                        pool.append(cid)
            if pool:
                self.rng.shuffle(pool)
                for cid in pool[:deficit]:
                    rr_keys = list(self.per_image_rel_pairs[cid].keys())
                    if not rr_keys:
                        continue
                    rr = rr_keys[self.rng.integers(len(rr_keys))]
                    rel_pairs_rr = self.per_image_rel_pairs[cid].get(rr, [])
                    if not rel_pairs_rr:
                        continue
                    pair_rr = rel_pairs_rr[self.rng.integers(len(rel_pairs_rr))]
                    shuf_ctx_ids.append(cid)
                    shuf_ctx_pairs.append(pair_rr)
                    used_ids.add(cid)

        if len(shuf_ctx_ids) != self.num_context:
            return None

        shuffled_image_ids = shuf_ctx_ids + [test_img_id]
        shuffled_pairs = shuf_ctx_pairs + [test_pair]

        return {
            "relation": relation,
            "task_id": task_id,
            "image_ids": chosen_ids,
            "pairs": pairs,
            "shuffled_image_ids": shuffled_image_ids,
            "shuffled_pairs": shuffled_pairs,
            "test_image": os.path.basename(self._resolve_image_path(test_img_id)),
        }

    def list_relations(self) -> List[str]:
        return sorted(self.tasks.keys())

    def set_relation(self, relation: Optional[str]) -> None:
        """
        Set the active relation for indexing. If None, indexing is over all tasks (flattened).
        """
        if relation is not None and relation not in self.tasks:
            raise ValueError(f"Relation '{relation}' not found. "
                             f"Available: {sorted(self.tasks.keys())}")
        self._current_relation = relation

    def get_relation(self) -> Optional[str]:
        """Return the currently active relation (or None if not set)."""
        return self._current_relation

    def set_shuffle_mode(self, use_shuffled: bool = True):
        """Setter to change default behavior at runtime."""
        self.use_shuffled = bool(use_shuffled)

    def set_matched_list(self, matched_list):
        matched_rel = matched_list['relation']
        matched_task_id = matched_list['task_id']
        filtered_tasks: Dict[str, List[Dict[str, Any]]] = {}
        for rel in self.tasks:
            filtered_tasks[rel] = []
            for item in self.tasks[rel]:
                if (rel, item["task_id"]) in list(zip(matched_rel, matched_task_id)):
                    filtered_tasks[rel].append(item)
        self.tasks = filtered_tasks
        print(f"============= Read {len(matched_task_id)} matched list, filtered to {self.__len__()} tasks ==============")

    def __len__(self) -> int:
        if getattr(self, "_current_relation", None):
            return len(self.tasks.get(self._current_relation, []))
        # flattened total
        return sum(len(v) for v in self.tasks.values())

    def get_len_for_relation(self, relation: str) -> int:
        return len(self.tasks.get(relation, []))

    def _resolve_image_path(self, image_id: str) -> str:
        return str(self.img_dir / f"{image_id}{self.image_ext}")

    def __getitem__(self, idx: int):
        """
        Get one ICL task:
          - If current relation is set, index within that relation's task list.
          - Else, index into the flattened (relation-ordered) task list.

        Returns (dict):
          {
            "images": List[Tensor or PIL],          # len = num_context + 1
            "question": str,                           # "s1:o1, s2:o2, ..., sK:oK, s_{K+1}:"
            "relation": str,                         # e.g., "above"
            "answer": str,                           # the expected final object (o_{K+1})
          }
        """
        # Pick the task
        if getattr(self, "_current_relation", None):
            rel = self._current_relation
            task = self.tasks[rel][idx]
        else:
            # Flatten in a deterministic order of relations
            if not hasattr(self, "_rel_order"):
                self._rel_order = sorted(self.tasks.keys())
            remaining = idx
            task = None
            for rel in self._rel_order:
                lst = self.tasks[rel]
                if remaining < len(lst):
                    task = lst[remaining]
                    break
                remaining -= len(lst)
            if task is None:
                raise IndexError("Index out of range for flattened tasks.")

        relation = task["relation"]
        image_ids = task["shuffled_image_ids"] if self.use_shuffled else task["image_ids"]
        pairs = task["shuffled_pairs"] if self.use_shuffled else task["pairs"]

        if self.output_zero_shot:
            images = [self._resolve_image_path(image_ids[-1])]
            question = construct_question([], pairs[self.num_context][0], self.model_config, images)
        else:
            # Load images
            images = [self._resolve_image_path(image_id) for image_id in image_ids]

            # Build prompt string: "s1:o1, s2:o2, ..., sK:oK, s_{K+1}:"
            question = construct_question(pairs[:self.num_context], pairs[self.num_context][0], self.model_config, images)

        return {
            "relation": relation,
            "task_id": task["task_id"],
            "images": images,
            "question": question,
            "answer": pairs[self.num_context][1],
            "pairs": pairs,
        }

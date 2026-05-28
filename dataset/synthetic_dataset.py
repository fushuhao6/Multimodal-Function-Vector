import os
import ast
from pathlib import Path
from typing import List, Tuple, Dict, Any, Callable, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from dataset.utils import construct_question


# ---------------- Helper functions ----------------

def separate_context_images_and_objects(context_images):
    images = []
    objs = []
    if isinstance(context_images, str):
        context_images = ast.literal_eval(context_images)
    for image, obj1, obj2 in context_images:
        images.append(image)
        objs.append([obj1, obj2])
    return images, objs

def get_test_question(row):
    relation = row['relation']
    subject = manual_correct_obj_name(row['anchor'])
    obj = manual_correct_obj_name(row[f'{relation}_object'])
    return subject, obj


def manual_correct_obj_name(obj_name):
    if obj_name.startswith('antique'):
        return obj_name.replace('antique', '')
    elif obj_name == 'coffee':
        return 'mug'
    elif obj_name == 'greenbottle':
        return 'bottle'
    elif obj_name == 'baseballbat':
        return 'bat'
    return obj_name


# --------------- Dataset ----------------

class SyntheticRelationDataset(Dataset):
    """
    Dataset built from a metadata.csv with columns:
      - image_name
      - anchor
      - above_object, below_object, left_object, right_object

    Args:
      data_root: directory containing 'metadata.csv' (images not loaded)
      num_context: number of in-context images to sample per item
      model_config: dict specifying the model family (e.g., 'internvl' or 'flamingo')
      seed: base seed for deterministic NumPy sampling
      use_shuffled: default context type if __getitem__ doesn't override
    """
    def __init__(
        self,
        data_root: str,
        num_context: int,
        model_config: Dict[str, Any],
        seed: int = 1234,
        use_shuffled: bool = False,
        diagonal: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.num_context = int(num_context)
        self.model_config = model_config
        self.use_shuffled = bool(use_shuffled)
        self.diagonal = bool(diagonal)

        # Read metadata
        meta_path = self.data_root / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.csv not found at {meta_path}")
        df = pd.read_csv(meta_path).reset_index(drop=True)

        self.df = df
        if not self.diagonal:
            self.relations = ['above', 'below', 'left', 'right']
        else:
            self.relations = ['above_left', 'above_right', 'below_left', 'below_right']
        self.name_to_row = {n: i for i, n in enumerate(df["image_name"])}
        self._current_relation: Optional[str] = None
        self._rel_order = self.relations[:]  # stable order for flattening

        # Precompute both train_images and shuffle_images (names only) with NumPy RNG
        self.items: List[Dict[str, Any]] = []
        for i, row in self.df.iterrows():
            rng = np.random.default_rng(self._row_seed(seed, i))

            relation_id = i % len(self.relations)
            relation = self.relations[relation_id]

            all_names = self.df["image_name"].tolist()
            this_name = row["image_name"]
            pool = [n for n in all_names if n != this_name]
            if len(pool) < self.num_context:
                raise ValueError(
                    f"Not enough images to sample {self.num_context} contexts (available {len(pool)})"
                )
            in_context = rng.choice(pool, size=self.num_context, replace=False).tolist()

            # --- train_images: relation-specific tuples (path, anchor, rel_obj) ---
            train_triplets: List[Tuple[str, str, str]] = []
            rel_col = f"{relation}_object"
            for name in in_context:
                r = self.df.loc[self.name_to_row[name]]
                anchor_obj = r["anchor"]
                rel_obj = r[rel_col]
                train_triplets.append((os.path.join(str(self.data_root), name), anchor_obj, rel_obj))

            # --- shuffle_images: zip in_context with rotating relation columns, then shuffle ---
            rel_cols = [f"{r}_object" for r in self.relations]
            shuffle_triplets: List[Tuple[str, str, str]] = []
            for name, obj_col in zip(in_context, rel_cols):
                r = self.df.loc[self.name_to_row[name]]
                anchor_obj = r["anchor"]
                obj = r[obj_col]
                shuffle_triplets.append((os.path.join(str(self.data_root), name), anchor_obj, obj))
            idx_perm = rng.permutation(len(shuffle_triplets))
            shuffle_triplets = [shuffle_triplets[j] for j in idx_perm]

            test_image = os.path.join(str(self.data_root), os.path.basename(this_name))

            item_row = {
                "image_name": this_name,
                "anchor": row["anchor"],
                "relation": relation,
            }
            for rel in self.relations:
                item_row[f"{rel}_object"] = row[f"{rel}_object"]

            self.items.append({
                "relation": relation,
                "relation_id": relation_id,
                "test_image": test_image,
                "train_images": train_triplets,    # list of (path, anchor, rel_obj)
                "shuffle_images": shuffle_triplets,# list of (path, anchor, obj)
                "row": item_row,
            })

        # Fast lookup: indices per relation
        self._idx_by_rel: Dict[str, List[int]] = {r: [] for r in self.relations}
        for i, it in enumerate(self.items):
            self._idx_by_rel[it["relation"]].append(i)

    def set_matched_list(self, matched_list):
        matched_rel = matched_list["relation"]
        matched_image = matched_list["image"]
        filtered_items: List[Dict[str, Any]] = []
        for item in self.items:
            if (item['relation'], os.path.basename(item['test_image'])) in list(zip(matched_rel, matched_image)):
                filtered_items.append(item)
        self.items = filtered_items

        # rebuild idx_by_rel
        self._idx_by_rel: Dict[str, List[int]] = {r: [] for r in self.relations}
        for i, it in enumerate(self.items):
            self._idx_by_rel[it["relation"]].append(i)
        print(f"============= Read {len(matched_rel)} matched list, filtered to {self.__len__()} tasks ==============")

    def __len__(self) -> int:
        """Length depends on whether a relation is active."""
        if self._current_relation is None:
            return len(self.items)
        return len(self._idx_by_rel[self._current_relation])

    def get_len_for_relation(self, relation: str) -> int:
        return len(self._idx_by_rel[relation])

    def list_relations(self) -> List[str]:
        return self.relations

    def set_relation(self, relation: Optional[str]) -> None:
        """
        Set the active relation for indexing. If None, indexing is over all items (flattened).
        """
        if relation is not None and relation not in self.relations:
            raise ValueError(f"Relation '{relation}' not found. Available: {self.relations}")
        self._current_relation = relation

    def get_relation(self) -> Optional[str]:
        """Return the currently active relation (or None if not set)."""
        return self._current_relation

    def _row_seed(self, base_seed: int, idx: int) -> int:
        return (int(base_seed) * 31 + int(idx)) & 0xFFFF_FFFF

    def set_shuffle_mode(self, use_shuffled: bool = True):
        """Setter to change default behavior at runtime."""
        self.use_shuffled = bool(use_shuffled)

    def construct_question_from_pairs(self, pairs, images=None):
        question = construct_question(pairs[:-1], pairs[-1][0], self.model_config, images)
        return question, pairs[-1][1]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetch one item. If 'shuffled' is None, uses self.use_shuffled.
        If a relation is set via set_relation(), idx is taken within that relation's slice.
        Otherwise, idx is over all items (flattened).

        Returns:
          {
            relation,
            images,                       # context_images + [test_image]
            question,                     # constructed prompt
            answer,                       # object for the test question
          }
        """
        # Resolve the *global* item index based on current relation
        if self._current_relation is None:
            global_idx = idx
            if not (0 <= global_idx < len(self.items)):
                raise IndexError("Index out of range.")
        else:
            rel_list = self._idx_by_rel[self._current_relation]
            if not (0 <= idx < len(rel_list)):
                raise IndexError(f"Index out of range for relation '{self._current_relation}'.")
            global_idx = rel_list[idx]

        # ---- existing per-item logic continues unchanged below ----
        it = self.items[global_idx]
        row = it["row"]

        relation = it['relation']
        test_image = it['test_image']

        # Choose context source (per-call arg overrides dataset default)
        use_shuf = self.use_shuffled
        src_key = 'shuffle_images' if use_shuf else 'train_images'

        context_images, img_descriptions = separate_context_images_and_objects(it[src_key])
        images = context_images + [test_image]

        # Build question
        test_question = get_test_question(row)
        question = construct_question(img_descriptions, test_question[0], self.model_config, images)

        return {
            "relation": relation,
            "images": images,
            "question": question,
            "answer": test_question[1],
            "pairs": img_descriptions + [test_question],
        }

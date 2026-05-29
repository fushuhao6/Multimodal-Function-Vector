# GQA Dataset Preparation

This repository uses a filtered subset of the GQA dataset for multimodal function vector experiments.

The pipeline consists of:

1. Downloading raw GQA data (images + scene graphs)
2. Filtering scene graphs into a structured dataset
3. Splitting into train/test sets for experiments

---

# Requirements

* Python ≥ 3.8
* numpy
* pillow (PIL)
* torch
* tqdm (optional, for progress bars)

---

# Step 0: Download GQA Dataset

Download the official GQA dataset from:

[https://cs.stanford.edu/people/dorarad/gqa/download.html](https://cs.stanford.edu/people/dorarad/gqa/download.html)

You will need:

* `train_sceneGraphs.json`
* `val_sceneGraphs.json`
*  GQA images

---

## Directory Structure

After downloading, organize files as follows:

```txt
data/
└── gqa/
    ├── raw/
    │   ├── train_sceneGraphs.json
    │   ├── val_sceneGraphs.json
    │   └── images/
    │       ├── 2353893.jpg
    │       ├── 2983410.jpg
    │       └── ...
```

# Step 1: Create Filtered GQA Dataset

Run:

```bash
python dataset/prepare_gqa_dataset.py \
    --relations "holding" "carrying" "riding" "sitting on" \
    --output_name gqa_agentic
```

---

## Output

```txt
data/gqa/processed/gqa_agentic.json
```

---

## What this step does

This script filters GQA scene graphs using:

* object size constraints
* object blacklist filtering
* object count constraints
* relation filtering (user-defined)
* relation ambiguity constraints

Output format:

```json
[
  {
    "image_id": "...",
    "objects": [...],
    "relations": [...]
  }
]
```

---

# Step 2: Train/Test Split

Run:

```bash
python dataset/split_gqa_dataset.py \
    --input data/gqa/processed/gqa_agentic.json \
    --split_strategy greedy \
    --seed 0
```

---

## Outputs

```txt
data/gqa/processed/gqa_agentic_train.json
data/gqa/processed/gqa_agentic_test.json
data/gqa/processed/gqa_agentic_split_metadata.json
```

---

## Split options

### Strategy options:

* `greedy` → balances relation distribution (recommended)
* `random` → simple 50/50 shuffle split

---

# Dataset Guarantees

This pipeline ensures:

* No image overlap between train and test
* Relation distributions are approximately balanced (greedy mode)
* All objects satisfy filtering constraints
* Only selected relations are included

---

# Example Workflow

```bash
# 1. Filter dataset
python dataset/prepare_gqa_dataset.py \
    --relations holding riding \
    --output_name experiment_1

# 2. Split dataset
python dataset/split_gqa_dataset.py \
    --input data/gqa/processed/experiment_1.json
```
# Multimodal Function Vectors for Visual Relations

> **ICML 2026** | [Paper](#citation)

Large Multimodal Models (LMMs) exhibit strong in-context learning from few multimodal demonstrations, yet the internal mechanisms behind this ability remain poorly understood. This repository accompanies our paper, which investigates how visual relational knowledge is encoded within LMMs.

We show that a small subset of attention heads is responsible for transmitting visual relation representations. The activations of these heads — termed **multimodal function vectors** — can be extracted, injected at inference time to improve zero-shot accuracy, and fine-tuned on modest training data (with LMM parameters frozen) to significantly outperform in-context learning baselines. We further show that relation-specific function vectors can be **linearly combined** to solve analogy problems involving novel, unseen visual relations.

Experiments are conducted on [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) and [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct). This codebase provides scripts for:

- **Causal mediation analysis** — identifying attention heads that drive relational predictions
- **Function vector extraction and evaluation** — computing and injecting function vectors for zero-shot inference
- **Function vector fine-tuning** — optimizing function vectors on a small labeled dataset
- **Generalization evaluation** — testing linear composition of function vectors on novel visual relations

---

## 1. Installation

We recommend using [Anaconda](https://www.anaconda.com/) (**Python 3.9+**). Each supported model requires its own conda environment, as described below.

### 1.1 Open-Flamingo

```bash
git clone https://github.com/mlfoundations/open_flamingo.git
cd open_flamingo
conda env create -f environment.yml
conda activate openflamingo
pip install tqdm pandas huggingface_hub
pip install git+https://github.com/davidbau/baukit seaborn transformers==4.28.1 numpy==1.26.4
```

### 1.2 Qwen3-VL

Install PyTorch first, then:

```bash
conda create -n qwen
conda activate qwen
pip install tqdm pandas huggingface_hub seaborn matplotlib accelerate "transformers>=4.57.0"
pip install git+https://github.com/davidbau/baukit
pip install --no-build-isolation flash-attn
```

> **Troubleshooting `flash-attn` on Ubuntu 20.04 or older:** Try this [known fix](https://github.com/modular/modular/issues/3684#issuecomment-2480409734). If the issue persists, pin to a specific version:
> ```bash
> pip install --no-build-isolation flash-attn==2.7.4.post1
> ```
> Alternatively, download a prebuilt wheel matching your environment from [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases).

---

## 2. Dataset Preparation

### 2.1 Synthetic Dataset

Run the following command to generate the synthetic dataset. It produces four subsets:

| Subset | Size | Purpose | Output folder |
|--------|------|---------|---------------|
| Eval | 4,000 images | Extracting function vectors | `data/synthetic_spatial_relation_eval` |
| Train | 1,000 images | Fine-tuning function vectors | `data/synthetic_spatial_relation_train` |
| Test | 1,000 images | Evaluation | `data/synthetic_spatial_relation_test` |
| Novel objects | 1,000 images | Object generalization (10 held-out objects) | `data/synthetic_spatial_relation_novel_objects_test` |

```bash
python dataset/generate_synthetic_images.py
```

### 2.2 GQA Dataset

We use a filtered subset of the [GQA dataset](https://cs.stanford.edu/people/dorarad/gqa/) focused on visual relational structure.

#### Step 1: Download GQA

Download the following from the official GQA release:

* GQA images (`images.zip` or full image directory)

For agentic and spatial relations, the data is already split into test and train json files.

After extraction, organize the dataset as (example from agentic relations):

```
data/
└── gqa/
    └── agentic/
        ├── gqa_filtered_agentic_test.json
        ├── gqa_filtered_agentic_train.json
        └── images/
            ├── <image_id>.jpg
            ├── <image_id>.jpg
            └── ...
```

#### Step 2: Using the dataset in experiments

The resulting JSON files are compatible with:

* `GQADataset` (object/relation supervision)
* `ICLRelationDataset` (in-context learning experiments)

Images are loaded from:

```
data/gqa/raw/images/
```

## 3. Adding a New Model

To support a new model, make the following changes:

1. Register the model in `utils/model_utils.py` → `load_model_and_tokenizer`.
2. Define how to construct prompts for the model in `dataset/utils.py` → `construct_question`.
3. Implement the model's forward pass in `utils/extract_utils.py` → `inference_model`.
4. Add answer token extraction logic in `utils/utils.py` → `get_answer_id`.
5. Handle swapped activations for the model in `utils/intervention_utils.py` → `rep_act`.
6. Register the model's output projection layer in `utils/extract_utils.py` → `compute_function_vector`.
7. *(For fine-tuning)* Add training example construction in `utils/utils.py` → `construct_training_labels`.

---

## 4. Running the Pipeline

### 4.0 (Optional) Test In-Context Learning

Verify that a model supports in-context learning before running the full pipeline:

```bash
python test_in_context_learning.py \
  -n <NUM_CONTEXT_EXAMPLES> \
  --model <MODEL_NAME> \
  --data_root data/synthetic_spatial_relation_test
```

Example: `-n 4 --model Qwen3-VL-4B-Instruct`

---

### 4.1 Compute Indirect Effect

Compute the indirect effect of each attention head at each layer:

```bash
python compute_indirect_effect.py \
  --model flamingo \
  --dataset synthetic \
  --data_root data/synthetic_spatial_relation_eval
```

*(Optional)* Visualize the results as a heatmap:

```bash
python plot_heatmap.py --model flamingo --dataset synthetic
```

---

### 4.2 Compute and Evaluate Function Vectors

Extract function vectors from the top-AIE heads and evaluate them on zero-shot tasks:

```bash
python eval_function_vector.py \
  --model flamingo \
  --dataset synthetic \
  --data_root data/synthetic_spatial_relation_test
```

Inspect the output to identify the best layer for injecting the function vector; you will need this (`LAYER_ID`) in the next steps.

---

### 4.3 Fine-Tune Function Vectors

Fine-tune the function vector for a given spatial relation on the training split:

```bash
python opt_function_vector.py \
  --model flamingo \
  --relation <RELATION> \
  --edit_layer <LAYER_ID> \
  --dataset synthetic \
  --train_root data/synthetic_spatial_relation_train \
  --test_root data/synthetic_spatial_relation_eval
```

---

### 4.4 Evaluate Fine-Tuned Function Vectors

Compare the baseline and fine-tuned function vectors on the test split:

```bash
python eval_opted_function_vector.py \
  --model flamingo \
  --relation <RELATION> \
  --edit_layer <LAYER_ID> \
  --dataset synthetic \
  --data_root data/synthetic_spatial_relation_test
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{fu2026multimodal,
  title     = {Multimodal Function Vectors for Visual Relations},
  author    = {Fu, Shuhao and Goldberg, Esther and Wu, Ying Nian and Lu, Hongjing},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026}
}
```
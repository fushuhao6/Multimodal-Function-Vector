import os
import torch
import random
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_images_flamingo(images, image_processor):
    ims = [Image.open(im) for im in images]

    vision_x = [image_processor(im).unsqueeze(0) for im in ims]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    return vision_x


def get_answer_id(query, answer, tokenizer, processor=None, model_config=None):
    """
    Parameters:
    query (str): query as a string
    answer (str): expected answer as a string
    tokenizer: huggingface tokenizer

    Returns:
    answer_ids (list): A list of the contextualized tokens of the answer
    """
    if processor is None:
        source = tokenizer(query, truncation=False, padding=False).input_ids
        target = tokenizer(query + answer, truncation=False, padding=False).input_ids
        assert len(source) < len(target) < tokenizer.model_max_length
        answer_ids = target[len(source):]
    else:
        if "qwen3" in model_config['name_or_path'].lower():
            inputs = processor.apply_chat_template(
                query,
                tokenize=True,
                add_generation_prompt=True,  # stops right before assistant’s turn
                return_tensors="pt"
            )
            # Tokenize full sequence with the answer appended
            full_msgs = query + [
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            full = processor.apply_chat_template(
                full_msgs,
                tokenize=True,
                add_generation_prompt=True,  # stops right before assistant’s turn
                return_tensors="pt"
            )
            # Determine where the answer starts
            answer_start = inputs.shape[1]
            answer_ids = full[0, answer_start:]
    return answer_ids


def construct_training_labels(question, answer, model_config, tokenizer, processor, device=None):
    """
    Construct labels where loss is applied ONLY on the first answer token.

    Parameters
    ----------
    question : str or list[dict]
        - For Flamingo: plain text string (your old format).
        - For LLaVA/Qwen/Gemma: chat-style messages list compatible with processor.apply_chat_template.
    answer : str
        Ground-truth answer text.
    model_config : dict
        Must contain 'name_or_path'.
    tokenizer : PreTrainedTokenizer
        For Flamingo branch.
    processor : Any
        For LLaVA/Qwen/Gemma branch (has apply_chat_template).
    device : torch.device or str, optional
        If given, returned tensors will be moved to this device.

    Returns
    -------
    inputs : dict
        Dict with at least "input_ids" (and others like attention_mask, pixel_values for VLMs).
    labels : torch.LongTensor
        Same shape as input_ids, with -100 everywhere except the first answer token.
    """
    name = model_config["name_or_path"].lower()

    # -------------------------
    # Flamingo-style models
    # -------------------------
    if "flamingo" in name:
        # question: string
        messages_train = f"{question}{answer}"

        tokenizer.padding_side = "left"

        # Tokenize question alone to get split index
        q_ids = tokenizer([question], return_tensors="pt")["input_ids"]
        colon_index = q_ids.shape[1] - 1  # index of last token in question

        # Tokenize question+answer for actual inputs
        lang_x = tokenizer([messages_train], return_tensors="pt")
        input_ids = lang_x["input_ids"]  # (1, L)

        # Labels: ignore everywhere, keep only first answer token (colon_index + 1)
        labels = torch.full_like(input_ids, -100)
        labels[:, colon_index + 1] = input_ids[:, colon_index + 1]

    # -------------------------
    # Chat-style VLMs (Qwen)
    # -------------------------
    elif any(keyword in name for keyword in ("qwen")):
        prompt_inputs = processor.apply_chat_template(
            question,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]  # first answer token index
        colon_index = prompt_len - 1

        messages_train = question + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    }
                ],
            }
        ]

        train_inputs = processor.apply_chat_template(
            messages_train,
            tokenize=True,
            add_generation_prompt=False,  # full convo with answer
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = train_inputs["input_ids"]

        # 3) Labels: loss only on the first answer token
        labels = torch.full_like(input_ids, -100)
        labels[:, prompt_len] = input_ids[:, prompt_len]
    else:
        raise ValueError(f"Unsupported model type for name_or_path={model_config['name_or_path']}")

    if device is not None:
        labels = labels.to(device)

    return messages_train, labels, colon_index


def find_topk_activations(data, topk=10):
    # Ensure data is a numpy array
    data = np.array(data)

    # Ensure data is 2D
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array. Got shape: {}".format(data.shape))

    # --- Find the top k values in the data ---
    flattened = data.flatten()
    topk_idx = np.argsort(flattened)[-topk:]

    # Convert flattened indices back to (row, col) coordinates
    n_layers, n_heads = data.shape
    topk_coords = [(idx // n_heads, idx % n_heads) for idx in topk_idx]

    return topk_coords


def show_heatmap_with_box(data, topk=10, figure_name='Heatmap of Heads vs. Layers', save_path=None):
    """
    Displays a heatmap of the given 2D array or DataFrame,
    and draws pink bounding boxes around the highest activations.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        A 2D array-like object with shape (n_layers, n_heads).
    """
    topk_coords = find_topk_activations(data, topk=topk)

    # Transpose data to flip axes (now rows: heads, columns: layers)
    data_flipped = np.array(data).T

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        data_flipped,
        cmap='bwr_r',
        center=0
    )

    # Draw pink bounding boxes
    for (layer, head) in topk_coords:
        rect = Rectangle(
            (layer, head), 1, 1,
            fill=False,
            edgecolor='#FF1493',
            linewidth=2
        )
        ax.add_patch(rect)

    # Set tick intervals and show labels
    x_ticks = np.arange(0, data.shape[0], 5)
    y_ticks = np.arange(0, data.shape[1], 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)

    plt.xlabel('Layer')
    plt.ylabel('Head Index')
    # plt.title(figure_name)  # Uncomment if title is needed

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def find_layer_id(layer_names):
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    layer_id = []
    for layer_name in layer_names:
        match = re.search(r'(layers|blocks)\.(\d+)', layer_name)
        assert match, f"No match found in: {layer_name}"
        layer_id.append(int(match.group(2)))
    return layer_id


def stuff_activations(activations, layers, max_layer):
    valid_layer_ids = find_layer_id(layers)
    new_activations = {rel: [] for rel in activations}
    count = 0
    for i in range(max_layer):
        if i in valid_layer_ids:
            for rel in activations:
                new_activations[rel].append(activations[rel][count])
            count += 1
        else:
            for rel in activations:
                new_activations[rel].append(None)
    return new_activations


def _ubuntu_version():
    """Return (major, minor) as ints, or (0, 0) if not Ubuntu/unknown."""
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            txt = f.read()
        if 'Ubuntu' not in txt:
            return (0, 0)
        m = re.search(r'VERSION_ID="?(?P<v>\d+\.\d+)"?', txt)
        if not m:
            return (0, 0)
        major, minor = m.group('v').split('.')
        return (int(major), int(minor))
    except Exception:
        return (0, 0)


def _is_ampere_or_newer():
    """True if at least one CUDA device has SM >= 80."""
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return (major, minor) >= (8, 0)
    except Exception:
        return False


def _flash_attn_available():
    """True if flash-attn is importable (FA2 installs CUDA extensions)."""
    try:
        # any of these imports indicates flash-attn 2 is present
        import flash_attn_2_cuda  # noqa: F401
        return True
    except Exception:
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
            return True
        except Exception:
            return False


def pick_attention_and_dtype():
    # Ubuntu gate: only auto-enable FlashAttention for Ubuntu > 20.04
    ubuntu_ok = _ubuntu_version() > (20, 4)
    ampere_ok = _is_ampere_or_newer()
    fa_ok = _flash_attn_available()

    # dtype: bf16 on Ampere+; else fp16 (bf16 isn’t supported on Turing)
    dtype = torch.bfloat16 if ampere_ok else torch.float16

    # attention backend
    if ubuntu_ok and ampere_ok and fa_ok:
        attn_impl = "flash_attention_2"
    else:
        # SDPA is fast and widely available in recent PyTorch/Transformers
        attn_impl = "sdpa"

    return attn_impl, dtype
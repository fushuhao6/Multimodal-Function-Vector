import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoProcessor
from huggingface_hub import hf_hub_download
from utils.utils import pick_attention_and_dtype


def load_model_and_tokenizer(model_name: str, device='cuda'):
    """
    Loads a huggingface model and its tokenizer

    Parameters:
    model_name: huggingface name of the model to load (e.g. GPTJ: "EleutherAI/gpt-j-6B", or "EleutherAI/gpt-j-6b")
    device: 'cuda' or 'cpu'

    Returns:
    model: huggingface model
    tokenizer: huggingface tokenizer
    MODEL_CONFIG: config variables w/ standardized names

    """
    assert model_name is not None
    processor = None
    image_processor = None
    tokenizer = None
    print("Loading: ", model_name)
    min_pixels = 256 * 28 * 28
    max_pixels = 256 * 28 * 28
    attn_impl, dtype = pick_attention_and_dtype()
    print(f"Using attn_implementation={attn_impl}, dtype={dtype}, cuda={torch.cuda.is_available()}")

    if 'flamingo' in model_name:
        from open_flamingo import create_model_and_transforms, Flamingo
        if '1b' in model_name:
            print("Loading mpt-1b-redpajama-200b-dolly")
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
                tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
                cross_attn_every_n_layers=1
            )

            checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt")
            model.load_state_dict(torch.load(checkpoint_path), strict=False)

            language_config = {
                    "n_heads": model.lang_encoder.config.n_heads,
                    "n_layers": model.lang_encoder.config.n_layers,
                    "resid_dim": model.lang_dim,
                    "attn_hook_names": [f'lang_encoder.transformer.blocks.{layer}.decoder_layer.attn.out_proj' for layer in
                                        range(model.lang_encoder.config.n_layers)],
                    "layer_hook_names": [f'lang_encoder.transformer.blocks.{layer}' for layer in
                                         range(model.lang_encoder.config.n_layers)],
                    "prepend_bos": False
            }
            cross_attn_config = {
                    "n_heads": model.lang_encoder.transformer.blocks[0].gated_cross_attn_layer.attn.heads,
                    "n_layers": model.lang_encoder.config.n_layers,
                    "resid_dim": model.lang_encoder.transformer.blocks[-1].gated_cross_attn_layer.attn.to_out.in_features,
                    "resid_out_dim": model.lang_encoder.transformer.blocks[-1].gated_cross_attn_layer.attn.to_out.out_features,
                    "attn_hook_names": [f'lang_encoder.transformer.blocks.{layer}.gated_cross_attn_layer.attn.to_out' for layer in
                                        range(model.lang_encoder.config.n_layers)],
                    "layer_hook_names": [f'lang_encoder.transformer.blocks.{layer}.gated_cross_attn_layer.attn' for layer in
                                        range(model.lang_encoder.config.n_layers)],
            }
        else:
            print("Loading RedPajama-INCITE-Instruct-3B-v1")
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
                tokenizer_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
                cross_attn_every_n_layers=2
            )

            checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct", "checkpoint.pt")
            model.load_state_dict(torch.load(checkpoint_path), strict=False)

            language_config = {
                    "n_heads": model.lang_encoder.gpt_neox.config.num_attention_heads,
                    "n_layers": model.lang_encoder.gpt_neox.config.num_hidden_layers,
                    "resid_dim": model.lang_dim,
                    "attn_hook_names": [f'lang_encoder.gpt_neox.layers.{layer}.decoder_layer.attention.dense' for layer in
                                        range(model.lang_encoder.gpt_neox.config.num_hidden_layers)],
                    "layer_hook_names": [f'lang_encoder.gpt_neox.layers.{layer}' for layer in
                                         range(model.lang_encoder.gpt_neox.config.num_hidden_layers)],
                    "attn_vis_hook_names": [f'lang_encoder.gpt_neox.layers.{layer}.decoder_layer.attention.dense' for layer in
                                        range(model.lang_encoder.gpt_neox.config.num_hidden_layers)],
                    "prepend_bos": False
            }
            cross_attn_config = {
                    "n_heads": model.lang_encoder.gpt_neox.layers[-1].gated_cross_attn_layer.attn.heads,
                    "n_layers": len([layer for layer in range(model.lang_encoder.gpt_neox.config.num_hidden_layers) if model.lang_encoder.gpt_neox.layers[layer].gated_cross_attn_layer is not None]),
                    "resid_dim": model.lang_encoder.gpt_neox.layers[-1].gated_cross_attn_layer.attn.to_out.in_features,
                    "resid_out_dim": model.lang_encoder.gpt_neox.layers[-1].gated_cross_attn_layer.attn.to_out.out_features,
                    "attn_hook_names": [f'lang_encoder.gpt_neox.layers.{layer}.gated_cross_attn_layer.attn.to_out' for layer in
                                        range(model.lang_encoder.gpt_neox.config.num_hidden_layers) if model.lang_encoder.gpt_neox.layers[layer].gated_cross_attn_layer is not None],
                    "layer_hook_names": [f'lang_encoder.gpt_neox.layers.{layer}.gated_cross_attn_layer.attn' for layer in
                                        range(model.lang_encoder.gpt_neox.config.num_hidden_layers) if model.lang_encoder.gpt_neox.layers[layer].gated_cross_attn_layer is not None],
            }

        model = model.to(device)
        model.device = next(model.parameters()).device

        MODEL_CONFIG = {
            "name_or_path": model_name,
            "language_model": language_config,
            "vision_model": {
                "n_heads": model.vision_encoder.transformer.resblocks[0].attn.num_heads,
                "n_layers": model.vision_encoder.transformer.layers,
                "resid_dim": model.vis_dim,
                "attn_hook_names": [f'vision_encoder.transformer.resblocks.{layer}.attn.out_proj' for layer in
                                    range(model.vision_encoder.transformer.layers)],
                "layer_hook_names": [f'vision_encoder.transformer.resblocks.{layer}' for layer in
                                     range(model.vision_encoder.transformer.layers)],
            },
            "perceiver": {
                "n_heads": model.perceiver.layers[0][0].heads,
                "n_layers": len(model.perceiver.layers),
                "resid_dim": model.perceiver.layers[0][0].to_out.in_features,
                "attn_hook_names": [f'perceiver.layers.{layer}.0.to_out' for layer in
                                    range(len(model.perceiver.layers))],
                "layer_hook_names": [f'perceiver.layers.{layer}' for layer in
                                     range(len(model.perceiver.layers))],
            },
            "cross_attn": cross_attn_config,
        }
    elif 'qwen' in model_name.lower():
        from transformers import Qwen3VLForConditionalGeneration
        model_name = "Qwen/Qwen3-VL-4B-Instruct"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation=attn_impl,
            device_map="auto",
        )
        resid_dim = model.config.text_config.num_attention_heads * model.config.text_config.head_dim

        processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model.eval()

        MODEL_CONFIG = {
            "name_or_path": model_name,
            "language_model": {
                "n_heads": model.config.text_config.num_attention_heads,
                "n_layers": model.config.text_config.num_hidden_layers,
                "resid_dim": resid_dim,
                "resid_out_dim": model.config.text_config.hidden_size,
                "attn_hook_names": [f'model.language_model.layers.{layer}.self_attn.o_proj' for layer in
                                    range(model.config.text_config.num_hidden_layers)],
                "layer_hook_names": [f'model.language_model.layers.{layer}' for layer in
                                     range(model.config.text_config.num_hidden_layers)],
                "attn_vis_hook_names": [f'model.language_model.layers.{layer}.self_attn.o_proj' for layer in
                                        range(model.config.text_config.num_hidden_layers)],
                "prepend_bos": False
            },
        }
    else:
        raise NotImplementedError("Still working to get this model available!")

    return model, tokenizer, MODEL_CONFIG, image_processor, processor

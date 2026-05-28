import os.path
import torch
import numpy as np
from baukit import TraceDict
from tqdm import tqdm
from utils.utils import get_answer_id, find_layer_id, read_images_flamingo
from utils.eval_utils import compute_individual_token_rank


# Attention Activations
def split_activations_by_head(activations, model_config):
    new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim'] // model_config[
        'n_heads'])  # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
    activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
    return activations.detach().to('cpu')


def get_task_conditioned_activation(dataset, model, tokenizer, model_config, image_processor, processor, submodel, topk=1):
    relations = dataset.list_relations()
    activation_storage = {r: [] for r in relations}
    matched_list = {'relation': [], 'task_id': [], 'image': []}

    eval_acc = 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
    for i, data in pbar:
        relation = data['relation']
        images = data['images']
        question = data['question']
        token_id_of_interest = get_answer_id(question, data['answer'], tokenizer, processor, model_config)[0]
        task_id = data['task_id'] if 'task_id' in data else None

        # stack_activation has shape (n_layers, n_heads, n_tokens, head_hidden_dim)
        stack_activation, output = get_head_activations(model, model_config, images, question, tokenizer,
                                                        image_processor=image_processor, processor=processor, submodel=submodel)

        rank = compute_individual_token_rank(output, token_id_of_interest)
        matched = rank < topk
        eval_acc += float(matched)
        pbar.set_description("Eval Acc: %.5f" % (eval_acc / (i + 1),))

        if matched:
            activation_storage[relation].append(stack_activation)
            matched_list['image'].append(os.path.basename(images[-1]))
            matched_list['relation'].append(relation)
            if task_id is not None:
                matched_list['task_id'].append(task_id)

        del output
        torch.cuda.empty_cache()

    mean_activations = {}
    for r in relations:
        print(f"Averaging {len(activation_storage[r])} activations for relation {r}")
        mean_activations[r] = torch.stack(activation_storage[r]).mean(dim=0)      # average over data, (n_layers, n_heads, head_hidden_dim)
        assert mean_activations[r].ndim >= 3
    return matched_list, mean_activations


def inference_model(model, tokenizer, model_config, images, question, image_processor=None,
                    processor=None, labels=None, layers=[],
                    retain_output=False, retain_input=False, retain_grad=False,
                    edit_output=None, use_cache=False, idx=None):

    device = model.device
    if labels is not None:
        labels = labels.to(device)

    # Prepare argument injector for model(...)
    def call_model(**kwargs):
        """Wrap model() so idx is passed only when not None."""
        if idx is not None:
            kwargs["idx"] = idx
        return model(**kwargs)

    with TraceDict(model, layers=layers, retain_input=retain_input,
                   retain_output=retain_output, retain_grad=retain_grad,
                   edit_output=edit_output) as activations_td:

        # -----------------------------------------------------------
        # Flamingo
        # -----------------------------------------------------------
        if "flamingo" in model_config['name_or_path'].lower():
            tokenizer.padding_side = "left"

            lang_x = tokenizer([question], return_tensors="pt")
            if labels is None:
                labels = lang_x["input_ids"].clone()
            vision_x = read_images_flamingo(images, image_processor)

            outputs = call_model(
                vision_x=vision_x.to(device),
                lang_x=lang_x["input_ids"].to(device),
                attention_mask=lang_x["attention_mask"].to(device),
                labels=labels,
                use_cache=use_cache,
            )

        # -----------------------------------------------------------
        # Qwen3
        # -----------------------------------------------------------
        elif "qwen3" in model_config["name_or_path"].lower():
            if labels is None:
                # INFERENCE: question is the usual chat (no assistant answer),
                inputs = processor.apply_chat_template(
                    question,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(device)
                labels = inputs.input_ids.clone()
            else:
                # TRAINING: question is messages_train from construct_training_labels
                inputs = processor.apply_chat_template(
                    question,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(device)

            outputs = call_model(
                **inputs,
                labels=labels,
                return_dict=True,
                use_cache=use_cache,
            )

        else:
            raise ValueError(f"Unrecognized model: {model_config['name_or_path']}")

    return outputs, activations_td



def get_head_activations(model, model_config, images, question, tokenizer, image_processor, processor, submodel):
    layers = model_config[submodel]['attn_hook_names']

    outputs, activations_td = inference_model(model, tokenizer, model_config, images, question,
                                              image_processor=image_processor, processor=processor, layers=layers, retain_input=True)
    logits = outputs.logits[:, -1, :].detach().cpu()        # (B, vocab_size)

    activations = [split_activations_by_head(activations_td[layer].input, model_config[submodel]) for layer in model_config[submodel]['attn_hook_names']]
    if submodel == 'perceiver':
        activations = [activation[:, -1, :, :] for activation in activations]       # (only take the last image)
    stack_initial = torch.vstack(activations).permute(0, 2, 1, 3)  # (n_layers, n_heads, n_tokens, head_hidden_dim)
    if submodel == 'perceiver':
        stack_filtered = stack_initial                          # (n_layers, n_heads, n_tokens, head_hidden_dim)
    else:
        stack_filtered = stack_initial[:, :, -1, :]           # Last token, (n_layers, n_heads, head_hidden_dim)
    return stack_filtered, logits



def compute_function_vector(mean_activations, mean_indirect_effect, model, submodel, model_config, n_top_heads=10):
    """
        Computes a "function vector" vector that communicates the task observed in ICL examples used for downstream intervention.

        Parameters:
        mean_activations: tensor of size (Layers, Heads, Tokens, head_dim) containing the average activation of each head for a particular task
        mean_indirect_effect: tensor of size (Layers, Heads) containing the indirect_effect of each layer and each head
        model: huggingface model being used
        model_config: contains model config information (n layers, n heads, etc.)
        n_top_heads: The number of heads to use when computing the summed function vector
        token_class_idx: int indicating which token class to use, -1 is default for last token computations

        Returns:
        function_vector: vector representing the communication of a particular task
        top_heads: list of the top influential heads represented as tuples [(L,H,S), ...], (L=Layer, H=Head, S=Avg. Indirect Effect Score)
    """
    model_resid_dim = model_config[submodel]['resid_dim']
    # if input_dim is different from output_dim
    if 'resid_out_dim' in model_config[submodel]:
        model_resid_out_dim = model_config[submodel]['resid_out_dim']
    else:
        model_resid_out_dim = model_resid_dim
    model_n_heads = model_config[submodel]['n_heads']
    model_head_dim = model_resid_dim // model_n_heads
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Compute Top Influential Heads (L,H)
    h_shape = mean_indirect_effect.shape
    topk_vals, topk_inds = torch.topk(mean_indirect_effect.view(-1), k=n_top_heads, largest=True)
    top_lh = list(zip(*np.unravel_index(topk_inds, h_shape), [round(x.item(), 4) for x in topk_vals]))
    top_heads = top_lh[:n_top_heads]

    # Compute Function Vector as sum of influential heads
    function_vector = torch.zeros((1, 1, model_resid_out_dim)).to(device)

    for L, H, _ in top_heads:
        if 'flamingo' in model_config['name_or_path'].lower():
            if submodel == 'language_model':
                out_proj = model.lang_encoder.gpt_neox.layers[L].decoder_layer.attention.dense
            elif submodel == 'cross_attn':
                layer_id = find_layer_id(model_config[submodel]['attn_hook_names'][L])[0]
                out_proj = model.lang_encoder.gpt_neox.layers[layer_id].gated_cross_attn_layer.attn.to_out
            else:
                raise NotImplementedError
        elif 'qwen' in model_config['name_or_path'].lower():
            if submodel == 'language_model':
                out_proj = model.model.language_model.layers[L].self_attn.o_proj
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        x = torch.zeros(model_resid_dim)
        x[H * model_head_dim:(H + 1) * model_head_dim] = mean_activations[L, H]
        d_out = out_proj(x.reshape(1, 1, model_resid_dim).to(device).to(dtype))

        function_vector += d_out.to(device)

    function_vector = function_vector.to(dtype)
    function_vector = function_vector.reshape(1, model_resid_out_dim)

    return function_vector, top_heads

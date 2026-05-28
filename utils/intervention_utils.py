from baukit import TraceDict, get_module
import torch
import re
import torch.nn.functional as F
from utils.utils import find_layer_id, read_images_flamingo


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_layer_idx(layer_name):
    try:
        return int(layer_name.split('.')[-1])
    except:
        match = re.search(r'(layers|blocks)\.(\d+)', layer_name)
        if match:
            return int(match.group(2))
        return -1


def replace_activation_w_avg(layer_head_pairs, avg_activations, model, model_config, submodel):
    """
    An intervention function for replacing activations with a computed average value.
    This function replaces the output of one (or several) attention head(s) with a pre-computed average value
    (usually taken from another set of runs with a particular property).
    The batched_input flag is used for systematic interventions where we are sweeping over all attention heads for a given (layer,token)
    The last_token_only flag is used for interventions where we only intervene on the last token (such as zero-shot or concept-naming)

    Parameters:
    layer_head_token_pairs: list of tuple triplets each containing a layer index, head index, and token index [(L,H,T), ...]
    avg_activations: torch tensor of the average activations (across ICL prompts) for each attention head of the model.
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    idx_map: dict mapping prompt label indices to ground truth label indices
    batched_input: whether or not to batch the intervention across all heads
    last_token_only: whether our intervention is only at the last token

    Returns:
    rep_act: A function that specifies how to replace activations with an average when given a hooked pytorch module.
    """
    edit_layers = [x[0] for x in layer_head_pairs]

    def rep_act(output, layer_name, inputs):
        current_layer = get_layer_idx(layer_name)
        assert current_layer >= 0
        if current_layer in edit_layers:
            inputs = inputs[0] if isinstance(inputs, tuple) else inputs

            # Determine shapes for intervention
            H = model_config[submodel]['n_heads']
            D = model_config[submodel]['resid_dim'] // H
            original_shape = inputs.shape

            # Make sure it's contiguous, then reshape; CLONE BEFORE any in-place edits
            inputs = inputs.contiguous().reshape(*inputs.size()[:-1], H, D).clone()

            # Perform Intervention:
            # Patch activations only at the last token for interventions like
            token_idx = -1 if submodel == 'language_model' else 0
            for (layer, head_n) in layer_head_pairs:
                if layer == current_layer:
                    if submodel == 'perceiver':
                        # avg_activations[layer][head_n] has shape (num_tokens, head_dim)
                        inputs[-1, -1, :, head_n] = avg_activations[layer][head_n]
                    else:
                        inputs[-1, token_idx, head_n] = avg_activations[layer][head_n]

            inputs = inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight

            name = model_config['name_or_path'].lower()
            if any(keyword in name for keyword in ("flamingo", "qwen")):
                inputs = inputs.squeeze()
                if inputs.ndim < 2:
                    inputs = inputs.unsqueeze(0)
                out_proj_bias = proj_module.bias
                if out_proj_bias is None:
                    new_output = torch.matmul(inputs, out_proj.T)
                else:
                    new_output = torch.addmm(out_proj_bias, inputs, out_proj.T)
            return new_output
        else:
            return output

    return rep_act


def add_function_vector(edit_layer, fv_vector, idx=-1):
    """
    Adds a vector to the output of a specified layer in the model

    Parameters:
    edit_layer: the layer to perform the FV intervention
    fv_vector: the function vector to add as an intervention
    device: device of the model (cuda gpu or cpu)
    idx: the token index to add the function vector at

    Returns:
    add_act: a fuction specifying how to add a function vector to a layer's output hidden state
    """

    def add_act(output, layer_name):
        current_layer = get_layer_idx(layer_name)
        if current_layer == edit_layer:
            if isinstance(output, tuple):
                output[0][:, idx] += fv_vector.to(output[0].device)
                return output
            else:
                output[:, idx] += fv_vector.to(output.device)
                return output
        else:
            return output

    return add_act


def get_word_rank(d_out, tokenizer, word):
    # Step 1: Get token ID
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"'{word}' maps to multiple tokens: {token_ids}")
    token_id = token_ids[0]

    # Step 2: Sort the entire distribution
    d_out = d_out.squeeze()  # shape: [vocab_size]
    sorted_vals, sorted_inds = torch.sort(d_out, descending=True)  # sorted_inds: token ids in ranked order

    # Step 3: Find rank of token_id
    rank = (sorted_inds == token_id).nonzero(as_tuple=True)[0].item() + 1  # +1 to make rank start at 1
    score = d_out[token_id].item()
    return rank, score


def fv_to_vocab_by_injection(function_vector, model, submodel, model_config, tokenizer, image_processor, relation, n_tokens=10):
    """
    Decodes a provided function vector into the model's vocabulary embedding space.

    Parameters:
    function_vector: torch vector extracted from ICL contexts that represents a particular function
    model: huggingface model
    model_config: dict with model information - n_layers, n_heads, etc.
    tokenizer: huggingface tokenizer
    n_tokens: number of top tokens to include in the decoding

    Returns:
    decoded_tokens: list of tuples of the form [(token, probability), ...]
    """
    device = model.device
    images = ['../results/spatial_relation_function_vector_test/test_16.png']
    vision_x = read_images_flamingo(images, image_processor)

    tokenizer.padding_side = "left"
    lang_x = tokenizer(["tape:"], return_tensors="pt")

    outputs = model(
        vision_x=vision_x.to(device),
        lang_x=lang_x["input_ids"].long().to(device),
        attention_mask=lang_x["attention_mask"].to(device),
    )
    outputs = outputs.logits[:, -1, :].detach().cpu()
    d_out = F.softmax(outputs, dim=-1)
    vals, inds = torch.topk(d_out, k=n_tokens, largest=True)
    clean_decoded_tokens = [(tokenizer.decode(x), round(y.item(), 4)) for x, y in zip(inds.squeeze(), vals.squeeze())]
    print(clean_decoded_tokens)

    # For every layer, head, token combination perform the replacement & track the change in meaningful tokens
    edit_layer = 14 if relation == 'topdown' else 17
    edit_layer_id = edit_layer
    if submodel == 'cross_attn':
        edit_layer_id = find_layer_id(model_config[submodel]['layer_hook_names'][edit_layer])[0]

    # Perform Intervention
    intervention_fn = add_function_vector(edit_layer_id, function_vector, idx=-1)

    with TraceDict(model, layers=model_config[submodel]['layer_hook_names'], edit_output=intervention_fn):
        outputs = model(
            vision_x=vision_x.to(device),
            lang_x=lang_x["input_ids"].long().to(device),
            attention_mask=lang_x["attention_mask"].to(device),
        )
    outputs = outputs.logits[:, -1, :].detach().cpu()
    d_out = F.softmax(outputs, dim=-1)

    vals, inds = torch.topk(d_out, k=n_tokens, largest=True)
    decoded_tokens = [(tokenizer.decode(x), round(y.item(), 4)) for x, y in zip(inds.squeeze(), vals.squeeze())]
    print(decoded_tokens)
    return decoded_tokens

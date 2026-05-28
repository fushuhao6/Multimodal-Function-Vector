import os
import torch
import argparse
import pickle
from tqdm import tqdm
from utils.utils import (set_seed, get_answer_id, find_layer_id, stuff_activations)
from utils.model_utils import load_model_and_tokenizer
from utils.extract_utils import inference_model, get_task_conditioned_activation
from utils.intervention_utils import replace_activation_w_avg
from dataset import build_relation_dataset


def get_indirect_effect_by_task(dataset, avg_activations, model, tokenizer, model_config, image_processor, processor, submodel, start_layer=0, max_correct_per_relation=-1, verbose=False):
    indirect_effect_storage = {}
    clean_acc = {}
    intervention_acc = {}
    count = {}
    for rel in dataset.list_relations():
        indirect_effect_storage[rel] = torch.zeros(model_config[submodel]['n_layers'], model_config[submodel]['n_heads'])
        clean_acc[rel] = 0
        intervention_acc[rel] = torch.zeros(model_config[submodel]['n_layers'], model_config[submodel]['n_heads'])
        count[rel] = 0

    if submodel == 'cross_attn':
        avg_activations = stuff_activations(avg_activations, model_config[submodel]['attn_hook_names'], model_config['language_model']['n_layers'])

    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        relation = data['relation']
        images = data['images']
        question = data['question']
        token_id_of_interest = get_answer_id(question, data['answer'], tokenizer, processor, model_config)[0]

        if count[relation] >= max_correct_per_relation > 0:
            continue

        if verbose:
            for image in images:
                print(os.path.basename(image))
            print(f"token_id_of_interest: {token_id_of_interest}")

        # Clean Run of Baseline:
        outputs, _ = inference_model(model, tokenizer, model_config, images, question,
                                     image_processor=image_processor, processor=processor)

        clean_output = outputs.logits[:, -1, :].detach().cpu()
        clean_probs = torch.softmax(clean_output[0], dim=-1)
        clean_acc[relation] += float(torch.argmax(clean_probs) == token_id_of_interest)

        if verbose:
            next_token_id = torch.argmax(clean_output, dim=-1)
            target_token_str = tokenizer.decode(token_id_of_interest)
            next_token_str = tokenizer.decode(next_token_id)
            print(f"prediction: {next_token_str}, label: {target_token_str}, correct: {float(torch.argmax(clean_probs) == token_id_of_interest)}")

        # For every layer, head, token combination perform the replacement & track the change in meaningful tokens
        for layer in range(start_layer, model_config[submodel]['n_layers']):
            head_hook_layer = [model_config[submodel]['attn_hook_names'][layer]]
            layer_id = layer
            if submodel == 'cross_attn':
                layer_id = find_layer_id(head_hook_layer)[0]

            for head_n in range(model_config[submodel]['n_heads']):
                intervention_locations = [(layer_id, head_n)]
                intervention_fn = replace_activation_w_avg(
                    layer_head_pairs=intervention_locations,
                    avg_activations=avg_activations[relation],
                    model=model,
                    model_config=model_config,
                    submodel=submodel
                )

                outputs, _ = inference_model(model, tokenizer, model_config, images, question,
                                             image_processor=image_processor, processor=processor, layers=head_hook_layer,
                                             edit_output=intervention_fn)

                # Move the output logits to CPU and free GPU memory held by outputs
                output = outputs.logits[:, -1, :].detach().cpu()  # batch_size x n_tokens x vocab_size, only want last token prediction

                # Delete outputs and any other GPU tensors if no longer needed
                del outputs
                torch.cuda.empty_cache()

                # TRACK probs of tokens of interest
                intervention_probs = torch.softmax(output, dim=-1)  # convert to probability distribution
                intervention_acc[relation][layer, head_n] += float(torch.argmax(intervention_probs) == token_id_of_interest)

                if verbose:
                    print(f'------------------------------------- Layer {layer} Head {head_n} --------------------------')
                    next_token_id = torch.argmax(output, dim=-1)
                    target_token_str = tokenizer.decode(token_id_of_interest)
                    next_token_str = tokenizer.decode(next_token_id)
                    print(f"prediction: {next_token_str} with prob {torch.max(intervention_probs)}, "
                          f"label: {target_token_str}, "
                          f"correct: {float(torch.argmax(intervention_probs) == token_id_of_interest)}")

                # For multiple token IDs, index_select returns a tensor with shape (batch_size, len(token_id_of_interest))
                selected_diff = (intervention_probs - clean_probs).index_select(
                    1, torch.LongTensor([token_id_of_interest]).squeeze()
                ).squeeze()

                indirect_effect_storage[relation][layer, head_n] += selected_diff

                # Optionally, delete any tensors not needed for the next iteration
                del output, intervention_probs, selected_diff
                torch.cuda.empty_cache()
        count[relation] += 1

    for rel in indirect_effect_storage.keys():
        indirect_effect_storage[rel] = indirect_effect_storage[rel] / count[rel]
        intervention_acc[rel] = intervention_acc[rel] / count[rel]
        clean_acc[rel] = clean_acc[rel] / count[rel]

    return indirect_effect_storage, clean_acc, intervention_acc


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Inference InternVL 2.5")

    # Add arguments
    parser.add_argument('-n', '--num_context', type=int, default=4, help="Number of context images")
    parser.add_argument('--seed', type=int, default=1234, help="Seed for the experiment")
    parser.add_argument('--n_per_relation', type=int, default=1000, help="How many trials for each relation in GQA dataset")
    parser.add_argument('--max_correct_per_relation', type=int, default=-1, help="How many trials to calculate CIE for each relation (Does not affect how many trials to get the activations)")
    parser.add_argument('--start_layer', type=int, default=0, help="Which layer to start calculate CIE with")
    parser.add_argument('--model', type=str, default='flamingo', help="Which model to use")
    parser.add_argument('--submodel', type=str, default='language_model', help="Use either language_model or vision_model")
    parser.add_argument('--dataset', type=str, default='synthetic', help="Use either synthetic dataset or real image dataset")
    parser.add_argument('--data_root', type=str, default='data/synthetic_spatial_relation_eval', help="data root for the dataset")
    parser.add_argument('--redo', action='store_true', default=False, help="Redo all the steps")
    parser.add_argument('--verbose', action='store_true', default=False, help="Output auxillary information")

    # Parse the arguments
    args = parser.parse_args()
    set_seed(args.seed)
    print(f"============= Testing {args.model}-{args.submodel} with {args.num_context} context images =============")

    model, tokenizer, model_config, image_processor, processor = load_model_and_tokenizer(args.model, device='cuda:0')
    assert args.submodel in model_config, f"Submodel {args.submodel} not available in model {args.model}"

    # Load data
    model_name = os.path.basename(args.model)
    save_root = f'results/{model_name}/{args.dataset}'
    save_path = f'{save_root}/{model_name}_context_{args.num_context}_matched_list.pkl'
    activation_save_path = f'{save_root}/activations/{model_name}_{args.submodel}_{args.num_context}context_task_conditioned_activations.pkl'
    os.makedirs(f'{save_root}/activations', exist_ok=True)

    dataset = build_relation_dataset(
        args.dataset,
        data_root=args.data_root,
        json_file=os.path.join(args.data_root, 'gqa_filtered_spatial_train.json'),
        num_context=args.num_context,
        model_config=model_config,
        seed=args.seed,
        tasks_per_relation=args.n_per_relation,
    )

    # each mean activation should be of shape (n_layers, n_heads, head_hidden_dim)
    if not os.path.exists(activation_save_path) or args.redo:
        print("============= Gathering task conditioned activations =============")
        matched_list, mean_activations = get_task_conditioned_activation(dataset, model, tokenizer, model_config, image_processor, processor, args.submodel)
        # save matched list
        matched_list['num_context'] = args.num_context
        with open(save_path, 'wb') as f:
            pickle.dump(matched_list, f)
        with open(activation_save_path, 'wb') as f:
            pickle.dump(mean_activations, f)
    else:
        print(f"============= Average activations already exist, reading from {activation_save_path} =============")
        with open(activation_save_path, 'rb') as f:
            mean_activations = pickle.load(f)
        with open(save_path, 'rb') as f:
            matched_list = pickle.load(f)

    dataset.set_matched_list(matched_list)
    dataset.set_shuffle_mode(True)

    print(f'============= Read {len(dataset)} examples for indirect effect evaluation =============')

    # Calculate indirect effect for each relation
    indirect_effect_storage, clean_acc, intervention_acc = get_indirect_effect_by_task(dataset, mean_activations, model,
                                                                                       tokenizer, model_config, image_processor,
                                                                                       processor, args.submodel, start_layer=args.start_layer,
                                                                                       max_correct_per_relation=args.max_correct_per_relation, verbose=args.verbose)

    acc = {'clean': clean_acc, 'intervention': intervention_acc}
    with open(activation_save_path.replace('_task_conditioned_activations', '_acc'), 'wb') as f:
        pickle.dump(acc, f)

    with open(activation_save_path.replace('_task_conditioned_activations', '_indirect_effect'), 'wb') as f:
        pickle.dump(indirect_effect_storage, f)

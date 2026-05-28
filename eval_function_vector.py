import os
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import set_seed, get_answer_id, find_layer_id
from utils.model_utils import load_model_and_tokenizer
from utils.extract_utils import compute_function_vector, inference_model
from utils.intervention_utils import add_function_vector
from utils.eval_utils import compute_individual_token_rank, decode_to_vocab
from dataset import build_relation_dataset


def zero_shot_inference(dataset, model, tokenizer, model_config, image_processor, processor, topk=1, verbose=False):
    clean_acc = 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
    for i, data in pbar:
        relation = data['relation']
        images = data['images']
        question = data['question']
        answer = data['answer']
        token_id_of_interest = get_answer_id(question, answer, tokenizer, processor, model_config)[0]

        if verbose:
            for image in images:
                print(os.path.basename(image))
            print(f"token_id_of_interest: {token_id_of_interest}")

        # Clean Run of Baseline:
        outputs, _ = inference_model(model, tokenizer, model_config, images, question,
                                     image_processor=image_processor, processor=processor)
        clean_output = outputs.logits[:, -1, :].detach().cpu()

        target_rank = compute_individual_token_rank(clean_output, token_id_of_interest)

        matched = target_rank < topk
        clean_acc += float(matched)

        # update tqdm bar with accuracy so far
        running_acc = clean_acc / (i + 1)
        pbar.set_postfix(acc=f"{running_acc:.3f}")

        if verbose:
            top_k_response = decode_to_vocab(clean_output, tokenizer, k=topk)
            print(f'User: {question}\nTop {topk} Response: {top_k_response}\nLabel: {answer}\nMatched: {matched}')

    clean_acc = clean_acc / len(dataset)

    return clean_acc


def function_vector_intervention(dataset, function_vector, edit_layer, model, tokenizer, model_config, image_processor, processor, submodel, topk=1, verbose=False):
    intervention_acc = 0

    if 'resid_out_dim' in model_config[submodel]:
        model_resid_out_dim = model_config[submodel]['resid_out_dim']
    else:
        model_resid_out_dim = model_config[submodel]['resid_dim']

    function_vector = function_vector.reshape(1, model_resid_out_dim)

    pbar = tqdm(enumerate(dataset), total=len(dataset))
    for i, data in pbar:
        relation = data['relation']
        images = data['images']
        question = data['question']
        answer = data['answer']
        token_id_of_interest = get_answer_id(question, answer, tokenizer, processor, model_config)[0]

        # For every layer, head, token combination perform the replacement & track the change in meaningful tokens
        edit_layer_id = edit_layer
        if submodel == 'cross_attn':
            edit_layer_id = find_layer_id(model_config[submodel]['layer_hook_names'][edit_layer])[0]

        # Perform Intervention
        intervention_fn = add_function_vector(edit_layer_id, function_vector, idx=-1)
        outputs, _ = inference_model(model, tokenizer, model_config, images, question,
                                     image_processor=image_processor, processor=processor,
                                     layers=model_config[submodel]['layer_hook_names'], edit_output=intervention_fn)

        # Move the output logits to CPU and free GPU memory held by outputs
        output = outputs.logits[:, -1, :].detach().cpu()  # batch_size x n_tokens x vocab_size, only want last token prediction

        # TRACK probs of tokens of interest
        intervention_rank = compute_individual_token_rank(output, token_id_of_interest)
        matched = intervention_rank < topk
        intervention_acc += float(matched)

        # update tqdm bar with accuracy so far
        running_acc = intervention_acc / (i + 1)
        pbar.set_postfix(acc=f"{running_acc:.3f}")

        if verbose:
            top_k_response = decode_to_vocab(output, tokenizer, k=topk)
            prob = torch.softmax(output, dim=-1)[0]
            print(f'------------------------------------- Layer {layer} Function Vector Intervention --------------------------')
            print(f'User: {question}\nTop {topk} Response: {top_k_response}\nLabel: {answer}, output prob: {prob[token_id_of_interest]}\nMatched: {matched}')

        # delete any tensors not needed for the next iteration
        del output, outputs
        torch.cuda.empty_cache()
    intervention_acc = intervention_acc / len(dataset)

    return intervention_acc


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Inference InternVL 2.5")

    # Add arguments
    parser.add_argument('-n', '--num_context', type=int, default=4, help="Number of context images")
    parser.add_argument('--seed', type=int, default=1234, help="Seed for the experiment")
    parser.add_argument('--topk', type=int, default=1, help="Top k predictions.")
    parser.add_argument('--n_per_relation', type=int, default=1000, help="How many trials for each relation in GQA dataset")
    parser.add_argument('--model', type=str, default='flamingo', help="Which model to use")
    parser.add_argument('--start_layer', type=int, default=0, help="Which layer to start injecting function vectors")
    parser.add_argument('--submodel', type=str, default='language_model', help="Use either language_model or vision_model")
    parser.add_argument('--dataset', type=str, default='synthetic', help="Use either synthetic dataset or real image dataset")
    parser.add_argument('--data_root', type=str, default='data/synthetic_spatial_relation_eval', help="data root for the dataset")
    parser.add_argument('--verbose', action='store_true', default=False, help="Output auxillary information")

    # Parse the arguments
    args = parser.parse_args()

    set_seed(args.seed)
    model = args.model
    model_name = os.path.basename(model)

    model, tokenizer, model_config, image_processor, processor = load_model_and_tokenizer(model, device='cuda:0')

    result_root = f'results/{model_name}/{args.dataset}'
    matched_list_file = f'{result_root}/{model_name}_context_{args.num_context}_matched_list.pkl'
    activation_save_path = f'{result_root}/activations/{model_name}_{args.submodel}_{args.num_context}context_task_conditioned_activations.pkl'
    indirect_effect_file = activation_save_path.replace('_task_conditioned_activations', '_indirect_effect')
    acc_file = activation_save_path.replace('_task_conditioned_activations', '_acc')
    dataset = build_relation_dataset(
        args.dataset,
        data_root=args.data_root,
        num_context=0 if args.dataset == 'synthetic' else args.num_context,
        json_file=os.path.join(args.data_root, 'gqa_filtered_spatial_train.json'),
        model_config=model_config,
        seed=args.seed,
        output_zero_shot=True,
        tasks_per_relation=args.n_per_relation,
    )

    with open(matched_list_file, 'rb') as f:
        matched_list = pickle.load(f)
    dataset.set_matched_list(matched_list)

    with open(activation_save_path, 'rb') as f:
        mean_activations = pickle.load(f)

    save_root = f'{result_root}/plots'
    os.makedirs(save_root, exist_ok=True)

    with open(indirect_effect_file, 'rb') as f:
        indirect_effect = pickle.load(f)

    relations = dataset.list_relations()
    layers = list(range(args.start_layer, model_config[args.submodel]['n_layers']))
    n_top_head_list = [10]

    intervention_acc = {rel: torch.zeros(len(n_top_head_list), len(layers)) for rel in relations}
    zero_shot_acc = {rel: 0. for rel in relations}

    print('=============== Running Zero-Shot Inference ================')

    for key in relations:
        dataset.set_relation(key)
        zero_shot_acc[key] = zero_shot_inference(dataset, model, tokenizer, model_config, image_processor, processor,
                                                 topk=args.topk, verbose=args.verbose)

    print('=============== Evaluating Function Vectors on Zero-Shot Inference ================')
    for n_i, n_top_heads in enumerate(n_top_head_list):
        for key in relations:
            effect = indirect_effect[key]
            function_vector, top_heads = compute_function_vector(mean_activations[key], effect, model, args.submodel, model_config, n_top_heads=n_top_heads)
            if args.verbose:
                print(function_vector.shape)
                print(function_vector)
                print(f"Max of function vector: {function_vector.max():.04f}; min of function vector: {function_vector.min():.04f}")

            dataset.set_relation(key)
            for l_i, layer in enumerate(layers):
                intervention_acc[key][n_i, l_i] = function_vector_intervention(dataset, function_vector, layer, model, tokenizer,
                                                                              model_config, image_processor=image_processor,
                                                                              processor=processor, submodel=args.submodel,
                                                                              topk=args.topk, verbose=args.verbose)

            print(f"-------------- Finished relation: {key}, {n_top_heads} heads ------------------------")
            print(f"Zero-shot accuracy: {zero_shot_acc[key]:.4f}\nFunction vector eval accuracy: {intervention_acc[key][n_i]}")
            print("=" * 60)

            # Plot accuracy across each layer
            if len(layers) > 1:
                zero_shot_acc_array = np.ones_like(intervention_acc[key][n_i].numpy()) * zero_shot_acc[key]

                # Plotting
                plt.figure(figsize=(8, 5))
                plt.plot(layers, intervention_acc[key][n_i], label='Func Vec Accuracy', marker='o', color='green')
                plt.plot(layers, zero_shot_acc_array, label='Zero-Shot Accuracy', marker='s', color='orange')
                plt.xlabel('Layer', fontsize=20)
                plt.ylabel('Accuracy', fontsize=20)
                plt.xticks(layers)  # explicitly setting the x-axis ticks
                plt.yticks(fontsize=16)
                plt.title("")
                plt.legend()
                plt.grid(True)
                plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(os.path.join(save_root, f'fv_acc_{args.submodel}_{args.num_context}contexts_{n_top_heads}heads_top{args.topk}_{key}.png'))
                plt.show()

    # plot accuracy across difference number of heads
    if len(n_top_head_list) > 1:
        for key in relations:
            for l_i, layer in enumerate(layers):
                zero_shot_acc_array = np.ones_like(intervention_acc[key][:, l_i].numpy()) * zero_shot_acc[key]

                # Plotting
                plt.figure(figsize=(12, 5))
                plt.plot(n_top_head_list, intervention_acc[key][:, l_i], label='Func Vec Accuracy', marker='o', color='green')
                plt.plot(n_top_head_list, zero_shot_acc_array, label='Zero-Shot Accuracy', marker='s', color='orange')
                plt.xlabel('Number of top heads', fontsize=20)
                plt.ylabel('Accuracy', fontsize=20)
                plt.xticks(n_top_head_list)  # explicitly setting the x-axis ticks
                plt.yticks(fontsize=16)
                plt.title("")
                plt.legend()
                plt.grid(True)
                plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_root, f'fv_acc_{args.submodel}_{args.num_context}contexts_top{args.topk}_{key}_layer{layer}.png'))
                plt.show()

    accuracies = {"zero_shot_acc": zero_shot_acc, "intervention_acc": intervention_acc}

    csv_save_path = os.path.join(save_root, f'fv_acc_{args.submodel}_{args.num_context}contexts_heads_{n_top_head_list[0]}_to_{n_top_head_list[-1]}_layer_{layers[0]}_to_{layers[-1]}.pkl')
    with open(csv_save_path, 'wb') as f:
        pickle.dump(accuracies, f)

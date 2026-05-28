import os
import torch
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import set_seed
from utils.model_utils import load_model_and_tokenizer
from utils.extract_utils import compute_function_vector
from utils.opt_util import FunctionVectorModel, eval_zero_shot_fv
from dataset import build_relation_dataset
from test_in_context_learning import test_in_context_learning


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Evaluate finetuned function vectors")

    # Add arguments
    parser.add_argument('-n', '--num_context', type=int, default=4, help="Number of context images")
    parser.add_argument('--num_test_context', type=int, default=0, help="Number of context images")
    parser.add_argument('--topk', type=int, default=1, help="Top k predictions.")
    parser.add_argument('--gpu', type=int, default=0, help="which GPU to use.")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed.")
    parser.add_argument('--edit_layer', type=int, default=18, help="Which layer to insert function vector.")
    parser.add_argument('--opt_layer', type=int, default=None, help="Which layer to insert opted function vector.")
    parser.add_argument('--n_top_heads', type=int, default=10, help="Random seed.")
    parser.add_argument('--model', type=str, default='flamingo', help="Which model to use")
    parser.add_argument('--relation', type=str, default='above', help="Which relation")
    parser.add_argument('--submodel', type=str, default='language_model', help="Use either language_model or vision_model")
    parser.add_argument('--dataset', type=str, default='synthetic', help="Use either synthetic dataset or real image dataset")
    parser.add_argument('--data_root', type=str, default='data/synthetic_spatial_relation_test', help="testing data root for the dataset")
    parser.add_argument('--fv_path', type=str, default=None, help="path to the finetuned function vector. If not provided it will read from the default location.")
    parser.add_argument('--verbose', action='store_true', default=False, help="Output auxillary information")
    parser.add_argument('--save_results', type=bool, default=True, help="Whether to save current results")
    parser.add_argument('--n_per_relation', type=int, default=1000, help="How many trials for each relation in GQA dataset")

    # Parse the arguments
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    model = args.model
    model_name = os.path.basename(model)
    dataset = 'synthetic' if 'synthetic' in args.dataset else args.dataset        # workaround for synthetic novel
    result_root = f'results/{model_name}/{dataset}'
    activation_save_path = f'{result_root}/activations/{model_name}_{args.submodel}_{args.num_context}context_task_conditioned_activations.pkl'
    indirect_effect_file = activation_save_path.replace('_task_conditioned_activations', '_indirect_effect')
    fv_save_root = f'{result_root}/finetuned_function_vectors'

    if args.opt_layer is None:
        args.opt_layer = args.edit_layer

    plot_save_root = f'{result_root}/plots'
    os.makedirs(plot_save_root, exist_ok=True)

    model, tokenizer, model_config, image_processor, processor = load_model_and_tokenizer(model, device=device)

    test_dataset_with_context = build_relation_dataset(
        args.dataset,
        data_root=args.data_root,
        json_file=os.path.join(args.data_root, 'gqa_filtered_spatial_test.json'),
        num_context=args.num_context,
        model_config=model_config,
        seed=args.seed,
        tasks_per_relation=args.n_per_relation,
    )
    test_dataset_with_context.set_relation(args.relation)

    test_dataset = build_relation_dataset(
        args.dataset,
        data_root=args.data_root,
        json_file=os.path.join(args.data_root, 'gqa_filtered_spatial_test.json'),
        num_context=args.num_test_context,
        model_config=model_config,
        seed=args.seed,
        tasks_per_relation=args.n_per_relation,
    )
    test_dataset.set_relation(args.relation)

    with open(activation_save_path, 'rb') as f:
        mean_activations = pickle.load(f)

    with open(indirect_effect_file, 'rb') as f:
        indirect_effect = pickle.load(f)

    print(f"===================================== Relation: {args.relation} =====================================================")
    # zero-shot Flamingo
    zero_shot_acc = test_in_context_learning(test_dataset, model, tokenizer, model_config, image_processor, processor, verbose=args.verbose)

    # num_context Flamingo
    context_acc = test_in_context_learning(test_dataset_with_context, model, tokenizer, model_config, image_processor, processor, verbose=args.verbose)

    effect = indirect_effect[args.relation]
    function_vector, top_heads = compute_function_vector(mean_activations[args.relation], effect, model, args.submodel, model_config, n_top_heads=args.n_top_heads)
    if args.verbose:
        print(function_vector.shape)
        print(function_vector)
        print(f"Max of function vector: {function_vector.max():.04f}; min of function vector: {function_vector.min():.04f}")

    fv_model = FunctionVectorModel(model, args.edit_layer, function_vector, model_config, args.submodel)

    # initial function vector
    initial_fv_eval_acc = eval_zero_shot_fv(test_dataset, fv_model, tokenizer, image_processor, processor, model_config, verbose=args.verbose)

    # optimized function vector
    try:
        model_save_path = os.path.join(fv_save_root, f'finetuned_fv_{model_name}_{args.submodel}_{args.relation}.ckpt')
        if args.fv_path is not None:
            model_save_path = args.fv_path
        print(f"Loading function vector from {model_save_path}")
        function_vector = torch.load(model_save_path)
        fv_model.function_vector = function_vector
        fv_model.set_edit_layer(args.opt_layer)
        opt_fv_eval_acc = eval_zero_shot_fv(test_dataset, fv_model, tokenizer, image_processor, processor, model_config, verbose=args.verbose)
    except Exception as e:
        print(f"Error occurred: {e}. Setting opt_fv_eval_acc to 0...")
        opt_fv_eval_acc = 0

    print(f"{args.num_test_context} shot accuracy for {args.relation}: {zero_shot_acc:.4f}")
    print(f"{args.num_context}-shot accuracy for {args.relation}: {context_acc:.4f}")
    print(f"Initial FV zero-shot evaluation accuracy for {args.relation}: {initial_fv_eval_acc:.4f}")
    print(f"Optimized FV zero-shot valuation accuracy for {args.relation}: {opt_fv_eval_acc:.4f}")

    if args.save_results:
        # Data
        categories = [f'{args.num_test_context}-shot', f'{args.num_context}-shot', f'{args.num_test_context}-shot initial FV', f'{args.num_test_context}-shot fine-tuned FV']
        acc_values = [zero_shot_acc, context_acc, initial_fv_eval_acc, opt_fv_eval_acc]

        # ---- Save to CSV ----
        df = pd.DataFrame({
            "Category": categories,
            "Accuracy": acc_values
        })
        df.to_csv(os.path.join(plot_save_root, f"results_{args.submodel}_{args.dataset}_{args.relation}_{args.num_test_context}shot.csv"), index=False)

        custom_colors = ['#474747', '#999999', '#C08497', '#93C2F1']

        # Create bar plot
        plt.figure(figsize=(6, 5))
        plt.bar(categories, acc_values, color=custom_colors)

        plt.ylabel('Score')
        plt.ylim(0, 0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_root, f"results_{args.submodel}_{args.dataset}_{args.relation}_{args.num_test_context}shot.png"))

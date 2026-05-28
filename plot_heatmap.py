import os
import pickle
import argparse
from utils.utils import show_heatmap_with_box


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Inference InternVL 2.5")

    # Add arguments
    parser.add_argument('-n', '--num_context', type=int, default=4, help="Number of context images")
    parser.add_argument('--topk', type=int, default=1, help="Top k predictions.")
    parser.add_argument('--n_top_heads', type=int, default=10, help="Top k heads.")
    parser.add_argument('--dataset', type=str, default='synthetic',
                        help="Use either synthetic dataset or real image dataset")
    parser.add_argument('--model', type=str, default='flamingo', help="Which model to use")
    parser.add_argument('--submodel', type=str, default='language_model', help="Use either language_model or vision_model")
    parser.add_argument('--verbose', action='store_true', default=False, help="Output auxillary information")

    # Parse the arguments
    args = parser.parse_args()
    model_name = os.path.basename(args.model)
    result_root = f'results/{model_name}/{args.dataset}'
    activation_save_path = f'{result_root}/activations/{model_name}_{args.submodel}_{args.num_context}context_task_conditioned_activations.pkl'
    indirect_effect_file = activation_save_path.replace('_task_conditioned_activations', '_indirect_effect')
    acc_file = activation_save_path.replace('_task_conditioned_activations', '_acc')

    heatmap_save_root = f'{result_root}/plots'
    os.makedirs(heatmap_save_root, exist_ok=True)

    with open(acc_file, 'rb') as f:
        acc = pickle.load(f)

    with open(indirect_effect_file, 'rb') as f:
        indirect_effect = pickle.load(f)

    clean_acc = acc['clean']
    intervention_acc = acc['intervention']
    average_indirect_effect = []
    for key in clean_acc:
        acc_diff = intervention_acc[key] - clean_acc[key]
        save_path = os.path.join(heatmap_save_root, f"heatmap_accuracy_{args.model}_{args.submodel}_{key}.png")
        show_heatmap_with_box(acc_diff, topk=args.n_top_heads, figure_name=f"{args.model}_{args.submodel}-{key}, Accuracy difference", save_path=save_path)
        print(f"Average accuracy for {args.submodel}-{key}: {clean_acc[key]}, max acc difference: {acc_diff.max()}")

        save_path = os.path.join(heatmap_save_root, f"heatmap_AIE_{args.model}_{args.submodel}_{key}.png")
        show_heatmap_with_box(indirect_effect[key], topk=args.n_top_heads, figure_name=f"{args.model}_{args.submodel}-{key}, AIE", save_path=save_path)


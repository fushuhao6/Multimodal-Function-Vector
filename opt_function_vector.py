import os
import torch
import pickle
import argparse
from utils.utils import set_seed
from utils.model_utils import load_model_and_tokenizer
from utils.extract_utils import compute_function_vector
from utils.opt_util import FunctionVectorModel, zero_shot_opt_with_fv_model
from dataset import build_relation_dataset


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Inference InternVL 2.5")

    # Add arguments
    parser.add_argument('-n', '--num_context', type=int, default=4, help="Number of context images")
    parser.add_argument('--topk', type=int, default=1, help="Top k predictions.")
    parser.add_argument('--gpu', type=int, default=0, help="which GPU to use.")
    parser.add_argument('--epochs', type=int, default=10, help="Training epochs.")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate.")
    parser.add_argument('--edit_layer', type=int, default=18, help="Which layer to insert function vector.")
    parser.add_argument('--n_top_heads', type=int, default=10, help="Random seed.")
    parser.add_argument('--n_per_relation', type=int, default=1000, help="How many trials for each relation in GQA dataset")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed.")
    parser.add_argument('--model', type=str, default='flamingo', help="Which model to use")
    parser.add_argument('--relation', type=str, default='above', help="Which relation")
    parser.add_argument('--submodel', type=str, default='language_model', help="Use either language_model or vision_model")
    parser.add_argument('--dataset', type=str, default='synthetic', help="Use either synthetic dataset or real image dataset")
    parser.add_argument('--train_root', type=str, default='data/synthetic_spatial_relation_train', help="training data root for the dataset")
    parser.add_argument('--test_root', type=str, default='data/synthetic_spatial_relation_test', help="testing data root for the dataset")
    parser.add_argument('--verbose', action='store_true', default=False, help="Output auxillary information")
    parser.add_argument('--random_init', action='store_true', default=False, help="Randomly init function vector")

    # Parse the arguments
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    model = args.model
    model_name = os.path.basename(model)

    model, tokenizer, model_config, image_processor, processor = load_model_and_tokenizer(model, device=device)

    result_root = f'results/{model_name}/{args.dataset}'
    activation_save_path = f'{result_root}/activations/{model_name}_{args.submodel}_{args.num_context}context_task_conditioned_activations.pkl'
    indirect_effect_file = activation_save_path.replace('_task_conditioned_activations', '_indirect_effect')

    save_root = f'{result_root}/finetuned_function_vectors'
    os.makedirs(save_root, exist_ok=True)

    train_dataset = build_relation_dataset(
        args.dataset,
        data_root=args.train_root,
        json_file=os.path.join(args.train_root, 'gqa_filtered_spatial_train.json'),
        num_context=0,
        model_config=model_config,
        seed=args.seed,
        tasks_per_relation=args.n_per_relation,
    )
    test_dataset = build_relation_dataset(
        args.dataset,
        data_root=args.test_root,
        json_file=os.path.join(args.test_root, 'gqa_filtered_spatial_test.json'),
        num_context=0,
        model_config=model_config,
        seed=args.seed,
        tasks_per_relation=args.n_per_relation,
    )

    with open(activation_save_path, 'rb') as f:
        mean_activations = pickle.load(f)

    with open(indirect_effect_file, 'rb') as f:
        indirect_effect = pickle.load(f)

    train_dataset.set_relation(args.relation)
    test_dataset.set_relation(args.relation)

    print(f"============================ Relation: {args.relation} =================================")
    effect = indirect_effect[args.relation]
    function_vector, top_heads = compute_function_vector(mean_activations[args.relation], effect, model, args.submodel, model_config, n_top_heads=args.n_top_heads)
    if args.random_init:
        function_vector = torch.randn_like(function_vector)

    if args.verbose:
        print(function_vector.shape)
        print(function_vector)
        print(f"Max of function vector: {function_vector.max():.04f}; min of function vector: {function_vector.min():.04f}")

    fv_model = FunctionVectorModel(model, args.edit_layer, function_vector, model_config, args.submodel)
    raw_fv_save_path = os.path.join(save_root, f'raw_fv_{model_name}_{args.submodel}_{args.relation}.ckpt')
    torch.save(fv_model.function_vector, raw_fv_save_path)

    model_save_path = os.path.join(save_root, f'finetuned_fv_{model_name}_{args.submodel}_{args.relation}.ckpt')

    zero_shot_opt_with_fv_model(train_dataset, test_dataset, args.epochs, fv_model, tokenizer, image_processor, processor, model_config, lr=args.lr, path=model_save_path)



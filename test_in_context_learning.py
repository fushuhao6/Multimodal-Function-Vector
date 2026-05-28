import os
import torch
import argparse
from tqdm import tqdm
from utils.utils import (set_seed, get_answer_id)
from utils.model_utils import load_model_and_tokenizer
from utils.extract_utils import inference_model
from dataset import build_relation_dataset
from utils.eval_utils import compute_individual_token_rank


def test_in_context_learning(dataset, model, tokenizer, model_config, image_processor, processor, topk=1, verbose=False):
    eval_acc = 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
    for i, data in pbar:
        images = data['images']
        question = data['question']
        token_id_of_interest = get_answer_id(question, data['answer'], tokenizer, processor, model_config)[0]

        # stack_activation has shape (n_layers, n_heads, n_tokens, head_hidden_dim)
        outputs, _ = inference_model(model, tokenizer, model_config, images, question,
                                                  image_processor=image_processor, processor=processor)
        output = outputs.logits[:, -1, :].detach().cpu()  # (B, vocab_size)

        if verbose:
            print(question)
            next_token_id = torch.argmax(output, dim=-1)  # shape: [batch]
            target_token_str = tokenizer.decode(token_id_of_interest)
            next_token_str = tokenizer.decode(next_token_id)
            print(f"target tokens: {target_token_str}")
            print(f"next tokens: {next_token_str}")
            print("-" * 60)

        rank = compute_individual_token_rank(output, token_id_of_interest)
        matched = rank < topk
        eval_acc += float(matched)
        pbar.set_description("Eval Acc: %.5f" % (eval_acc / (i + 1),))

        del output
        torch.cuda.empty_cache()
    return eval_acc / len(dataset)


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Inference InternVL 2.5")

    # Add arguments
    parser.add_argument('-n', '--num_context', type=int, default=4, help="Number of context images")
    parser.add_argument('--seed', type=int, default=1234, help="Seed for the experiment")
    parser.add_argument('--model', type=str, default='flamingo', help="Which model to use")
    parser.add_argument('--dataset', type=str, default='synthetic', help="Use either synthetic dataset or real image dataset")
    parser.add_argument('--data_root', type=str, default='data/synthetic_spatial_relation_eval', help="data root for the dataset")
    parser.add_argument('--diagonal', action='store_true', default=False, help="diagonal dataset")
    parser.add_argument('--verbose', action='store_true', default=False, help="Output auxillary information")
    parser.add_argument('--n_per_relation', type=int, default=1000, help="How many trials for each relation in GQA dataset")

    # Parse the arguments
    args = parser.parse_args()
    set_seed(args.seed)
    print(f"============= Testing {args.model} with {args.num_context} context images =============")

    model, tokenizer, model_config, image_processor, processor = load_model_and_tokenizer(args.model, device='cuda:0')

    # Load data
    model_name = os.path.basename(args.model)

    dataset = build_relation_dataset(
        args.dataset,
        data_root=args.data_root,
        json_file=os.path.join(args.data_root, 'gqa_filtered_spatial_test.json'),
        num_context=args.num_context,
        model_config=model_config,
        seed=args.seed,
        diagonal=args.diagonal,
        tasks_per_relation=args.n_per_relation,
    )

    # each mean activation should be of shape (n_layers, n_heads, head_hidden_dim)
    eval_acc = test_in_context_learning(dataset, model, tokenizer, model_config, image_processor, processor, verbose=args.verbose)
    print(f"{args.model} Evaluation Accuracy on {args.dataset} with data root {os.path.basename(args.data_root)}: {eval_acc:.04f}")

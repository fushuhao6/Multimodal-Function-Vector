import numpy as np
import torch


def compute_individual_token_rank(prob_dist, target_id) -> int:
    """
    Individual computation of token ranks across a single distribution.

    Parameters:
    prob_dist: the distribution of scores for a single output
    target_id: the target id we care about

    Return:
    A single value representing the token rank for that single token
    """
    if isinstance(target_id, list):
        target_id = target_id[0]

    return torch.where(torch.argsort(prob_dist.squeeze(), descending=True) == target_id)[0].item()


def decode_to_vocab(prob_dist, tokenizer, k=5) -> list:
    """
    Decodes and returns the top K words of a probability distribution

    Parameters:
    prob_dist: torch tensor of model logits (distribution over the vocabulary)
    tokenizer: huggingface model tokenizer
    k: number of vocabulary words to include

    Returns:
    list of top K decoded vocabulary words in the probability distribution as strings, along with their probabilities (float)
    """
    get_topk = lambda x, K=1: torch.topk(torch.softmax(x, dim=-1), dim=-1, k=K)
    if not isinstance(prob_dist, torch.Tensor):
        prob_dist = torch.Tensor(prob_dist)

    return [(tokenizer.decode(x), round(y.item(), 5)) for x, y in
            zip(get_topk(prob_dist, k).indices[0], get_topk(prob_dist, k).values[0])]


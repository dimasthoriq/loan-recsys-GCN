"""
Author: Dimas Ahmad
Description: This file contains the evaluation functions for the project.
"""

import torch

def bpr_loss(positive, negative):
    diff = positive - negative
    ll = torch.mean(torch.nn.functional.logsigmoid(diff))
    return -ll


def ndcg_at_k(target, pred, k):
    _, indices = torch.topk(pred, k, dim=-1)

    ideal_relevance = torch.sort(target, descending=True).values[:k]
    predicted_relevance = target[indices]

    # Compute DCG
    gains = (2 ** predicted_relevance - 1) / torch.log2(torch.arange(2, k + 2).float().to(pred.device))
    dcg = torch.sum(gains)

    # Compute IDCG
    ideal_gains = (2 ** ideal_relevance - 1) / torch.log2(torch.arange(2, k + 2).float().to(pred.device))
    idcg = torch.sum(ideal_gains)

    # Avoid division by zero
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return ndcg.item()


def precision_at_k(target, pred, k):
    _, indices = torch.topk(pred, k, dim=-1)
    predicted_relevance = target[indices]
    precision = torch.sum(predicted_relevance) / k
    return precision.item()


def recall_at_k(target, pred, k):
    _, indices = torch.topk(pred, k, dim=-1)
    predicted_relevance = target[indices]
    recall = torch.sum(predicted_relevance) / torch.sum(target)
    return recall.item()

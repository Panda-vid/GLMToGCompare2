from multiprocessing import Pool
from typing import List
import torch
from torch.nn import functional as f

def token_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    assert predictions.shape == labels.shape, f"Tensor shapes do not agree between labels and prediciton. Labels shape {labels.shape}, predictions shape {predictions.shape}"
    total_size = torch.prod(torch.tensor(predictions.shape))
    total_overlap = torch.where(predictions == labels, 1, 0).sum()
    return total_overlap/total_size


def accuracy(predictions: List[str], labels: List[str]):
    total = len(predictions)
    with Pool(16) as pool:
        s = sum(pool.map(compare, zip(predictions, labels)))
    return s/total


def cross_entropy_manual(pred, groundtruth, ignore_index):
    target = groundtruth[groundtruth!= ignore_index]
    output = pred[groundtruth!=ignore_index]
    loss_manual = -1 * f.log_softmax(output, 1).gather(1, target.unsqueeze(1))
    loss_manual = loss_manual.mean()
    loss_manual.backward()
    return loss_manual
    

def compare(pred: str, label: str):
    return pred == label
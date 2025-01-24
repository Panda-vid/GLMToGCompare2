from multiprocessing import Pool
from typing import List, Tuple

import torch
import numpy as np

from torch.nn import functional as f

def token_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    assert predictions.shape == labels.shape, f"Tensor shapes do not agree between labels and prediciton. Labels shape {labels.shape}, predictions shape {predictions.shape}"
    total_size = torch.prod(torch.tensor(predictions.shape))
    total_overlap = torch.where(predictions == labels, 1, 0).sum()
    return total_overlap/total_size

def create_multiprocessed_accuracy(poolsize: int):
    pool = Pool(poolsize)
    def accuracy(predictions: List[str], labels: List[str]):
        total = len(predictions)
        s = sum(pool.map(compare, zip(predictions, labels)))
        return s/total
    return accuracy

def single_process_accuracy(predictions: List[str], labels: List[str]):
    return np.array([pred == label for (pred, label) in zip(predictions, labels)]).mean()


def cross_entropy_manual(pred, groundtruth, ignore_index):
    target = groundtruth[groundtruth!= ignore_index]
    output = pred[groundtruth!=ignore_index]
    loss_manual = -1 * f.log_softmax(output, 1).gather(1, target.unsqueeze(1))
    loss_manual = loss_manual.mean()
    loss_manual.backward()
    return loss_manual
    

def compare(pred_label_tuple: Tuple[str, str]):
    pred, label = pred_label_tuple
    return pred == label
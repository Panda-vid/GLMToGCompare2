import shutil

from multiprocessing import Pool
from typing import List, Tuple
from pathlib import Path

import torch
import numpy as np

from torch.nn import functional as f


class ModelCheckpoint:
    def __init__(self, location: Path):
        self.location = location
        self.convert_to_new_checkpoint_if_necessary()
        self.best_model_save = ModelSave(self.location / "best")
        self.latest_model_save = ModelSave(self.location / "latest")
        self.location.mkdir(exist_ok=True, parents=True)

    def convert_to_new_checkpoint_if_necessary(self):
        if self.location.exists():
            if self._is_old_checkpoint():
                self._convert_to_new_checkpoint()

    def _convert_to_new_checkpoint(self):
        encoder_location = self.location / "encoder"
        generator_location = self.location / "generator"
        progress_location = self.location / "progress"

        best_location = self.location / "best"
        latest_location = self.location / "latest"

        shutil.copytree(encoder_location, best_location / "encoder")
        shutil.copytree(generator_location, best_location / "generator")
        shutil.copyfile(progress_location, best_location / "progress")
        shutil.copytree(best_location, latest_location)

        shutil.rmtree(encoder_location)
        shutil.rmtree(generator_location)
        progress_location.unlink()

    def _is_old_checkpoint(self):
        children = [child.name for child in self.location.iterdir()]
        return "encoder" in children and "generator" in children and "progress" in children
    
    def save_best(self, encoder, generator, tokenizer, progress: Tuple[int, int]):
        self.best_model_save.save(encoder, generator, tokenizer, progress)

    def save_latest(self, encoder, generator, tokenizer, progress: Tuple[int, int]):
        self.latest_model_save.save(encoder, generator, tokenizer, progress)

    def save_latest_progress(self, progress: Tuple[int, int]):
        self.latest_model_save.save_progress(progress)
    
    def save_best_progress(self, progress: Tuple[int, int]):
        self.best_model_save.save_progress(progress)
    
    def load_progress(self):
        return self.latest_model_save.load_progress()

    def delete_progress(self):
        self.latest_model_save.delete_progress()


class ModelSave:
    def __init__(self, save_location: Path):
        self.location = save_location
        self.encoder_location = self.location / "encoder"
        self.generator_location = self.location / "generator"
        self.progress_location= self.location / "progress"
    
    def save(self, encoder, generator, tokenizer, progress: Tuple[int, int]):
        self.save_model(encoder, generator, tokenizer)
        self.save_progress(progress)

    def save_model(self, encoder, generator, tokenizer):
        self.location.mkdir(exist_ok=True, parents=True)
        encoder.save_pretrained(self.encoder_location, from_pt=True, safe_serialization=False)
        generator.save_pretrained(self.generator_location, from_pt=True, safe_serialization=False)
        tokenizer.save_pretrained(self.encoder_location, from_pt=True)

    def save_progress(self, progress: Tuple[int, int]):
        self.location.mkdir(exist_ok=True, parents=True)
        with self.progress_location.open("w") as file:
            file.write(str(progress))

    def load_progress(self):
        if self.progress_location.exists():
            with self.progress_location.open("r") as file:
                return eval(file.read())
        else:
            return [0, 0]

    def delete_progress(self):
        self.progress_location.unlink(missing_ok=True)


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
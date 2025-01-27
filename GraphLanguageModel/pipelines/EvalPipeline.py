import torch
import numpy as np

from pathlib import Path
from typing import Callable

from tqdm import tqdm

from torch.utils.data import DataLoader

from GraphLanguageModel.pipelines.data_handling import GraphDataset, create_collate_fn
from GraphLanguageModel.pipelines.recipies import ModelRecipe
from GraphLanguageModel.pipelines.util import create_multiprocessed_accuracy, single_process_accuracy


class EvalPipeline:
    def __init__(self, dataloader: DataLoader, score_func: Callable[[torch.Tensor, torch.Tensor], float], repetitions: int, 
                 tokenizer, encoder, generator, max_generation_len: int, data_name: str = "", device: str = "cpu") -> None:
        self.dataloader = dataloader
        self.data_name = data_name
        self.score_func = score_func
        self.repetitions = repetitions
        
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.generator = generator
        self.max_generation_len = max_generation_len
        self.device = device

    def eval(self):
        self.encoder.eval()
        self.generator.eval()
        scores = []
        for i in range(self.repetitions):
            score = self.eval_round()
            scores.append(score.cpu())
        scores = np.array(scores)
        return scores.mean(), scores.std()

    def eval_round(self):
        scores = []
        with tqdm(self.dataloader, postfix=f"Evaluating on {self.data_name}", total=len(self.dataloader)) as pbar:
            for (inputs, attention_mask), labels in pbar:
                with torch.no_grad():
                    outputs = self.generator.generate(encoder_outputs=self.encoder(**inputs), max_new_tokens=self.max_generation_len, early_stopping=True, attention_mask=attention_mask)
                    predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                    batch_score = torch.tensor(self.score_func(predictions, labels)).to(self.device)
                    scores.append(batch_score)
                    pbar.set_description_str(f"Score last batch: {batch_score}")

        all_scores = torch.tensor(scores, dtype=torch.float)
        return all_scores.mean()

    @property
    def data_processor(self):
        return self.encoder.data_processor
    

class Builder:
    def __init__(self) -> None:
        self.is_classification = None
        self.eval_data = None
        self.batch_size = 4
        self.model_recipe = None
        self.repetitions = 1
        self.score_func = single_process_accuracy

        self.reporting_interval = 5
        self.checkpointing_interval = 5
        self.device = "cpu"

    def is_classification_task(self, is_classification: bool):
        self.is_classification = is_classification
        return self
    
    def set_eval_data(self, eval_data: Path):
        self.eval_data = eval_data
        return self 

    def add_model_recipe(self, model_recipe: ModelRecipe):
        self.model_recipe = model_recipe
        return self
    
    def set_reporting_interval(self, reporting_interval: int):
        self.reporting_interval = reporting_interval
        return self

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        return self
    
    def set_repetitions(self, repetitions: int):
        self.repetitions = repetitions
        return self
    
    def set_score_function(self, score_function: Callable[[torch.Tensor, torch.Tensor], float]):
        self.score_func = score_function
        return self
    
    def set_device(self, device: str):
        self.device = device
        return self

    def build(self) -> EvalPipeline:
        if not self._buildable():
            raise ValueError(self._generate_error_msg())
        tokenizer, encoder, generator = self.model_recipe.build(self.device)
        dataloader = DataLoader(self._create_dataset(encoder.data_processor, tokenizer), batch_size=self.batch_size, 
                                collate_fn=create_collate_fn(self.device, encoder.data_processor, tokenizer, self.model_recipe.max_generation_len))
        return EvalPipeline(dataloader, self.score_func, self.repetitions, tokenizer, encoder, generator, self.model_recipe.max_generation_len, data_name=self.eval_data.name, device=self.device)

    def _buildable(self) -> bool:
        return self.is_classification is not None and self.model_recipe is not None
    
    def _create_dataset(self, data_processor, tokenizer):
        return GraphDataset(self.eval_data, data_processor, tokenizer, self.model_recipe.graph_encoder_strategy, self.is_classification)

    def _generate_error_msg(self) -> str:
        msg = ""
        if self.model_recipe is None:
            msg += "You have to add a model recipe to build the evaluation pipeline!\n"
        if self.is_classification is None:
            msg += "You have to specify whether the task is a classification task to build the evaluation pipeline!\n"
        return msg

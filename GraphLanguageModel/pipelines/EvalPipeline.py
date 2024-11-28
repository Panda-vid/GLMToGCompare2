import torch
import numpy as np

from pathlib import Path
from typing import Callable

from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader

from GraphLanguageModel.pipelines.data_handling import GraphDataset, create_collate_fn
from GraphLanguageModel.pipelines.recipies import ModelRecipe
from GraphLanguageModel.pipelines.util import accuracy


class EvalPipeline:
    def __init__(self, accelerator: Accelerator, dataloader: DataLoader, score_func: Callable[[torch.Tensor, torch.Tensor], float], repetitions: int, 
                 tokenizer, encoder, generator, max_generation_len: int, data_name: str = "") -> None:
        self.accelerator = accelerator
        self.dataloader = dataloader
        self.data_name = data_name
        self.score_func = score_func
        self.repetitions = repetitions
        
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.generator = generator
        self.max_generation_len = max_generation_len

    def eval(self):
        self.encoder.eval()
        self.generator.eval()
        scores = []
        for i in range(self.repetitions):
            scores.append(self.eval_round())
        scores = np.array(scores)
        return scores.mean(), scores.std()

    def eval_round(self):
        scores = []
        unwrapped_generator = self.accelerator.unwrap_model(self.generator)
        with tqdm(self.dataloader, postfix=f"Evaluating on {self.data_name}", total=len(self.dataloader), disable=(not self.accelerator.is_local_main_process)) as pbar:
            for inputs, labels in pbar:
                with torch.no_grad():
                    outputs = unwrapped_generator.generate(encoder_outputs=self.encoder(**inputs), max_new_tokens=self.max_generation_len)[:, :labels.shape[1]]
                    predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                batch_score = torch.tensor(self.score_func(predictions, labels)).cuda(device=self.accelerator.device)
                batch_scores = self.accelerator.gather(batch_score)

                if self.accelerator.is_local_main_process:
                    parallel_batch_score = batch_scores.mean()
                    scores.append(parallel_batch_score)
                    pbar.set_description_str(f"Score last batch: {parallel_batch_score}")
                    pbar.update(len(batch_scores))

        return torch.tensor(scores).mean()

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
        self.score_func = accuracy

        self.reporting_interval = 5
        self.checkpointing_interval = 5

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

    def build(self) -> EvalPipeline:
        if not self._buildable():
            raise ValueError(self._generate_error_msg())
        tokenizer, encoder, generator = self.model_recipe.build()
        dataloader = DataLoader(self._create_dataset(), batch_size=self.batch_size, 
                                collate_fn=create_collate_fn(encoder.data_processor, tokenizer, self.model_recipe.max_generation_len))
        return EvalPipeline(dataloader, self.score_func, self.repetitions, tokenizer, encoder, generator, self.model_recipe.max_generation_len, data_name=self.eval_data.name)

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

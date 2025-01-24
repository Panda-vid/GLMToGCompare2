import json
from pathlib import Path
from random import randint
from typing import Callable, Dict

import torch
import numpy as np

from tqdm import tqdm
from GraphLanguageModel.pipelines.recipies import ModelRecipe
from GraphLanguageModel.pipelines.util import accuracy


class EvalPipeline:
    def __init__(self, is_classification: bool, eval_data: Path, batch_size: int, 
                 score_func: Callable[[torch.Tensor, torch.Tensor], float], 
                 repetitions: int, tokenizer, encoder, generator,
                 max_generation_len: int, graph_encoder_strategy: str, device: str) -> None:
        self.is_classification = is_classification
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.score_func = score_func
        self.repetitions = repetitions
        
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.generator = generator
        self.max_generation_len = max_generation_len
        self.graph_encoder_strategy = graph_encoder_strategy

        self.device = device
        self.total = self._get_file_total()

    def _get_file_total(self):
        with self.eval_data.open("r") as data_file:
            total = len(data_file.readlines())
        return int(total/self.batch_size) + 1

    def batcher(self):
        with self.eval_data.open("r") as data_file:
            batch = []
            labels = []
            for i, instance in tqdm(enumerate(data_file)):
                inp, label = self._convert_data_instance(json.loads(instance))
                batch.append(inp)
                labels.append(label)
                if ((i + 1) % self.batch_size) == 0 or i + 1 == self.total:
                    inputs = self.data_processor.to_batch(data_instances=batch, tokenizer=self.tokenizer, max_seq_len = 512, device=self.device)
                    labels = self.tokenizer(labels, return_tensors="pt", padding=True).input_ids.to(device=self.device)[:, :-1]
                    yield inputs, labels
                    batch = []
                    labels = []

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
        with tqdm(self.batcher(), postfix=f"Evaluating on {self.eval_data.name}", total=self.total) as pbar:
            for inputs, labels in pbar:
                with torch.no_grad():
                    outputs = self.generator.generate(encoder_outputs=self.encoder(**inputs), output_scores=True, max_new_tokens=self.max_generation_len, early_stopping=True)[:, :labels.shape[1]]
                predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                del outputs
                labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                batch_score = self.score_func(predictions, labels)
                scores.append(batch_score)
                pbar.set_description_str(f"Score last batch: {batch_score}")
                torch.cuda.empty_cache()
        return torch.tensor(scores).mean()

    def _convert_data_instance(self, instance: Dict):
        graph = instance["graph"]
        label = "<pad><extra_id_0> " + instance["output"][0] if self.is_classification else instance["output"][0]
        text = None
        if "text" in instance.keys():
            text = self._convert_text_data(instance["text"])
        return self.data_processor.encode_graph(tokenizer=self.tokenizer, g=graph, text=text, how=self.graph_encoder_strategy), label
    
    def _convert_text_data(self, text_instance):
        return text_instance[randint(0, len(text_instance) - 1)] if isinstance(text_instance, list) else text_instance

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

    def set_device(self, device: str):
        self.device = device
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
        tokenizer, encoder, generator = self.model_recipe.build(self.device)
        return EvalPipeline(self.is_classification, self.eval_data, self.batch_size, 
                            self.score_func, self.repetitions, tokenizer, encoder, generator, 
                            self.model_recipe.max_generation_len, 
                            self.model_recipe.graph_encoder_strategy, self.device)

    def _buildable(self) -> bool:
        return self.is_classification is not None and self.model_recipe is not None
    
    def _generate_error_msg(self) -> str:
        msg = ""
        if self.model_recipe is None:
            msg += "You have to add a model recipe to build the evaluation pipeline!\n"
        if self.is_classification is None:
            msg += "You have to specify whether the task is a classification task to build the evaluation pipeline!\n"
        return msg

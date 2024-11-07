import json
from pathlib import Path
from random import randint, random, sample
from typing import Dict

import torch

from tqdm import tqdm
from GraphLanguageModel.pipelines.EvalPipeline import EvalPipeline
from GraphLanguageModel.pipelines.recipies import ModelRecipe, TrainRecipe
from GraphLanguageModel.pipelines.util import accuracy


class TrainPipeline:
    def __init__(self, is_classification: int, train_data: Path, num_epochs: int, 
                 batch_size: int, early_stopping: int, optimizer, neighborhood_size: int, 
                 loss, tokenizer, encoder, generator, max_generation_len: int, 
                 graph_encoder_strategy: str, checkpointing_interval: int, device: str, 
                 save_location: Path, eval_data: Path = None) -> None:
        self.is_classification = is_classification
        
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.optimizer = optimizer
        self.neighborhood_size = neighborhood_size
        self.loss = loss
        
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.generator = generator
        self.max_generation_len = max_generation_len
        self.graph_encoder_strategy = graph_encoder_strategy

        self.checkpointing_interval = checkpointing_interval
        self.device = device
        self.total = self._get_total()
        self.eval_pipeline = self._create_eval_pipeline(eval_data) if eval_data is not None else None

        self.save_location = save_location
        self.encoder_save_location = self.save_location / "encoder"
        self.generator_save_location = self.save_location / "generator"
        self.progress_location = self.save_location / "progress"
        self.abort = 0

        self._read_progress()

    def _get_total(self):
        with self.train_data.open("r") as data_file:
            total = len(data_file.readlines())
        return int(total/self.batch_size) + 1
    
    def _read_progress(self):
        if self.progress_location.exists():
            with self.progress_location.open("r") as file:
                self.progress = eval(file.read())
        else:
            self.progress = [0, 0]

    def _create_eval_pipeline(self, eval_data: Path):
        return EvalPipeline(self.is_classification, eval_data, self.batch_size * 2, accuracy, 1, self.tokenizer, self.encoder,
                            self.generator, self.max_generation_len, self.graph_encoder_strategy, self.device)

    def train(self):
        self.last_accuracy, _ = self.eval_pipeline.eval()
        print(f"Score before training: {self.last_accuracy}")
        for epoch in range(self.num_epochs):
            if epoch > self.progress[0]:
                self.encoder.train()
                self.generator.train()
                self.train_epoch()
                self._increment_early_stopping()
                self._increment_epoch_progress()
                self._save_progress()
        self._reset_progress()

    def train_epoch(self):
        self.running_loss = 0
        self.batches_since_report = 0
        with tqdm(enumerate(self.batcher()), postfix=f"Training {self.train_data.name}", total=self.total - (int(self.progress[1]/self.batch_size))) as pbar:
            for i, (inputs, labels) in pbar:
                self.optimizer.zero_grad()
                batch_loss = self.generator(encoder_outputs=self.encoder(**inputs), labels=labels).loss
                batch_loss.backward()
                self.optimizer.step()
                self.running_loss += batch_loss.item()
                self.batches_since_report += 1
                del batch_loss
                torch.cuda.empty_cache()
                pbar.set_description_str(f"Average loss last batch: {self.running_loss/self.batches_since_report}")
                self._save_if_necessary(i)
                
    
    def _increment_early_stopping(self):
        if self.eval_pipeline is not None:
            avg_score, _ = self.eval_pipeline.eval()
            print(f"Score after last epoch: {avg_score}")
            if self.last_accuracy > avg_score:
                self.abort += 1
            else:
                self.abort = 0
                self.last_accuracy = avg_score

    def _increment_epoch_progress(self):
        self.progress[0] = self.num_epochs if self.abort >= self.early_stopping else self.progress[0] + 1

    def _reset_progress(self):
        self.progress_location.unlink(missing_ok=True)
    
    def _save_progress(self):
        self.save_location.mkdir(exist_ok=True, parents=True)
        with self.progress_location.open("w") as file:
            file.write(str(self.progress))

    def batcher(self):
        with self.train_data.open("r") as data_file:
            batch = []
            labels = []
            for i, instance in enumerate(data_file, start=self.progress[1]):
                inp, label = self._convert_data_instance(json.loads(instance))
                batch.append(inp)
                labels.append(label)
                if ((i + 1) % self.batch_size) == 0 or i + 1 == self.total:
                    inputs = self.data_processor.to_batch(data_instances=batch, tokenizer=self.tokenizer, max_seq_len = 512, device=self.device)
                    labels = self.tokenizer(labels, return_tensors="pt", padding=True).input_ids.to(device=self.device)
                    yield inputs, labels
                    batch = []
                    labels = []
                self.progress[1] += 1
        self.progress[1] = 0

    def _save_if_necessary(self, data_index: int):
        if (data_index + 1) % self.checkpointing_interval == 0 or data_index + 1 == self.total:
            if self.eval_pipeline is not None:
                current_accuracy, _= self.eval_pipeline.eval()
                if current_accuracy > self.last_accuracy:
                    self._save_model()
                    self.last_accuracy = current_accuracy
            else:
                self._save_model()
            self._save_progress()

    def _convert_data_instance(self, instance: Dict):
        if len(instance["graph"]) > self.neighborhood_size:
            graph = [instance["graph"][0]] + sample(instance["graph"][1:], self.neighborhood_size)
        else:
            graph = instance["graph"]
        label = "<extra_id_0> " + instance["output"][0] if self.is_classification else instance["output"][0]
        text = None
        if "text" in instance.keys():
            text = self._convert_text_data(instance["text"])
        return self.data_processor.encode_graph(tokenizer=self.tokenizer, g=graph, text=text, how=self.graph_encoder_strategy), label
    
    def _save_model(self):
        self.save_location.mkdir(exist_ok=True, parents=True)
        self.encoder.save_pretrained(self.encoder_save_location, from_pt=True)
        self.tokenizer.save_pretrained(self.encoder_save_location, from_pt=True)
        self.generator.save_pretrained(self.generator_save_location, from_pt=True)
        
    def _convert_text_data(self, text_instance):
        return text_instance[randint(0, len(text_instance) - 1)] if isinstance(text_instance, list) else text_instance

    @property
    def data_processor(self):
        return self.encoder.data_processor
    

class Builder:
    def __init__(self) -> None:
        self.train_recipe = None
        self.model_recipe = None
        self.eval_data = None
        self.save_location = None

        self.checkpointing_interval = 5
        self.device = "cpu"

    def add_train_recipe(self, train_recipe: TrainRecipe):
        self.train_recipe = train_recipe
        return self

    def add_model_recipe(self, model_recipe: ModelRecipe):
        self.model_recipe = model_recipe
        return self
    
    def add_save_location(self, save_location: Path):
        self.save_location = save_location
        return self
    
    def set_eval_data(self, eval_data: Path):
        self.eval_data = eval_data
        return self

    def set_checkpointing_interval(self, chcekpointing_interval: int):
        self.checkpointing_interval = chcekpointing_interval
        return self

    def set_device(self, device: str):
        self.device = device
        return self

    def build(self) -> TrainPipeline:
        if not self._buildable():
            raise ValueError(self._generate_error_msg())
        self.tokenizer, self.encoder, self.generator = self.model_recipe.build(self.device)
        self.loss, self.optimizer = self.train_recipe.build(list(self.encoder.parameters()) + list(self.generator.parameters()))
        return TrainPipeline(self.train_recipe.is_classification, self.train_recipe.train_data, self.train_recipe.num_epochs, 
                             self.train_recipe.batch_size, self.train_recipe.early_stopping, self.optimizer,
                             self.train_recipe.neighborhood_size, self.loss, self.tokenizer, self.encoder, 
                             self.generator, self.model_recipe.max_generation_len, self.model_recipe.graph_encoder_strategy,
                            self.checkpointing_interval, self.device, save_location=self.save_location, eval_data=self.eval_data)
    
    def _buildable(self) -> bool:
        return self.train_recipe is not None or self.model_recipe is not None or self.save_location is not None
    
    def _generate_error_msg(self) -> str:
        msg = ""
        if self.save_location is None:
            msg += "You have to add a save_location to build the training pipeline!\n"
        if self.model_recipe is None:
            msg += "You have to add a model recipe to build the training pipeline!\n"
        if self.train_recipe is None:
            msg += "You have to add a train recipe to build the training pipeline!\n"
        return msg
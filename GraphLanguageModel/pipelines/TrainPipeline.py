import numpy as np
import torch
import gc

from pathlib import Path

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from GraphLanguageModel.pipelines.EvalPipeline import EvalPipeline
from GraphLanguageModel.pipelines.data_handling import GraphDataset, create_collate_fn
from GraphLanguageModel.pipelines.recipies import ModelRecipe, TrainRecipe
from GraphLanguageModel.pipelines.util import ModelCheckpoint, create_multiprocessed_accuracy, single_process_accuracy


class TrainPipeline:
    def __init__(self, num_epochs: int, batch_size: int, early_stopping: int, train_data: Path,
                 is_classification: bool, neighborhood_size: int,
                 optimizer, tokenizer, encoder, generator, graph_encoder_strategy: str, 
                 model_checkpoint: ModelCheckpoint, checkpointing_interval: int,
                 device: str = "cpu", eval_pipeline: EvalPipeline = None, has_loss_trace: bool = False) -> None:

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.train_data = train_data
        self.is_classification = is_classification
        self.neighborhood_size = neighborhood_size

        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.generator = generator
        self.graph_encoder_strategy = graph_encoder_strategy
        self.device = device

        self.checkpointing_interval = checkpointing_interval
        self.eval_pipeline = eval_pipeline
        self.data_name = train_data.name
        self.has_loss_trace = has_loss_trace

        self.model_checkpoint = model_checkpoint
        self.progress = self.model_checkpoint.load_progress()
        self.last_accuracy = 0
        self.abort = 0
        self.last_epoch_accuracy = 0

        self.dataloader = self._get_dataloader()

    def _get_dataloader(self):
        self.offset = self.progress[1] * self.batch_size
        dataset = GraphDataset(self.train_data, self.encoder.data_processor, self.tokenizer, self.graph_encoder_strategy,
                               self.is_classification, neighborhood_size=self.neighborhood_size, data_offset=self.offset)
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=create_collate_fn(), num_workers=16, pin_memory=True, shuffle=False)

    def train(self):
        if self.eval_pipeline is not None:
            self.last_epoch_accuracy, _ = self.eval_run()
            self.last_accuracy = self.last_epoch_accuracy
            print(f"Score before training: {self.last_epoch_accuracy}")
        while self.progress[0] < self.num_epochs:
            if self.abort < self.early_stopping:
                self.encoder.train()
                self.generator.train()
                self.train_epoch()
                self._post_epoch()    

    def _post_epoch(self):
        self._increment_early_stopping()
        self._increment_epoch_progress()
        self._save_progress()
        self.dataloader = self._get_dataloader()

    def train_epoch(self):
        with tqdm(enumerate(self.dataloader), postfix=f"Training {self.data_name}", total=len(self.dataloader)) as pbar:
            for i, (inputs, labels) in pbar:
                self.optimizer.zero_grad()
                inputs, attention_mask = self.data_processor.to_batch(data_instances=inputs, tokenizer=self.tokenizer, 
                                                                      device=self.device, max_seq_len=512, return_attention_mask=True)
                labels = self.tokenizer(labels, return_tensors="pt", padding=True).input_ids.to(self.device)
                batch_loss = self.generator(encoder_outputs=self.encoder(**inputs), labels=labels, attention_mask=attention_mask).loss
                batch_loss.backward()
                self.optimizer.step()
                self.progress[1] += 1
                pbar.set_description_str(f"Average loss last batch: {batch_loss}")
                self._save_checkpoint_if_necessary(i)
                self._save_loss_if_necessary(batch_loss.detach().item())

    def eval_run(self):
        return self.eval_pipeline.eval() if self.eval_pipeline is not None else (0, 0)    
    
    def _increment_early_stopping(self):
        current_accuracy = self._get_current_accuracy(print_accuracy=True, print_format="\nAccuracy after last epoch: {current_accuracy}")
        if self.last_epoch_accuracy > current_accuracy:
            self.abort += 1
        else:
            self.abort = 0
            self.last_epoch_accuracy = current_accuracy

    def _increment_epoch_progress(self):
        self.progress[0] += 1
        self.progress[1] = 0

    def _save_checkpoint_if_necessary(self, data_index: int):
        if (data_index + 1) % self.checkpointing_interval == 0 or data_index + 1 == len(self.dataloader):
            self._save_checkpoint()

    def _save_loss_if_necessary(self, loss: float):
        if self.has_loss_trace:
            loss_file = self.model_checkpoint.location / "losses.csv"
            if not loss_file.exists():
                with loss_file.open("w") as f:
                    f.write("epoch;batch;loss;last_dev_acc")
            if loss_file.exists():
                with loss_file.open("a") as f:
                    f.write(f"\n{self.progress[0]};{self.progress[1]};{loss};{self.last_accuracy}")

    def _save_checkpoint(self):
        self.model_checkpoint.save_latest(self.encoder, self.generator, self.tokenizer, self.progress)
        self._save_best_model_if_necessary()

    def _save_progress(self):
        self.model_checkpoint.save_latest_progress(self.progress)
        if self.eval_pipeline is None:
            self.model_checkpoint.save_best_progress(self.progress)

    def _save_best_model_if_necessary(self):
        current_accuracy = self._get_current_accuracy()
        if current_accuracy >= self.last_accuracy:
            self.model_checkpoint.save_best(self.encoder, self.generator, self.tokenizer, self.progress)
            self.last_accuracy = current_accuracy
    
    def _get_current_accuracy(self, print_accuracy: bool = True, print_format: str = "\nCurrent accuracy: {current_accuracy}"):
        current_accuracy = 1
        if self.eval_pipeline is not None:
            current_accuracy, _= self.eval_run()
            if print_accuracy:
                print(print_format.format(current_accuracy=current_accuracy))
        return current_accuracy

    @property
    def data_processor(self):
        return self.encoder.data_processor


class Builder:
    def __init__(self) -> None:
        self.train_recipe = None
        self.model_recipe = None
        self.eval_data = None
        self.save_location = None
        self.has_loss_trace = False
        self.device = "cpu"

        self.checkpointing_interval = 5

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
    
    def trace_loss(self, has_loss_trace: bool):
        self.has_loss_trace = has_loss_trace
        return self

    def build(self) -> TrainPipeline:
        if not self._buildable():
            raise ValueError(self._generate_error_msg())
        tokenizer, encoder, generator = self.model_recipe.build(self.device)
        optimizer = self.train_recipe.build(set(list(encoder.parameters()) + list(generator.parameters())))
        eval_pipeline = self._create_eval_pipeline(tokenizer, encoder, generator)
        return TrainPipeline(self.train_recipe.num_epochs, self.train_recipe.batch_size, self.train_recipe.early_stopping, self.train_recipe.train_data,
                             self.train_recipe.is_classification, self.train_recipe.neighborhood_size, optimizer, tokenizer, encoder, generator, 
                             self.model_recipe.graph_encoder_strategy, self.model_recipe.model_checkpoint, 
                             self.checkpointing_interval, device=self.device, eval_pipeline=eval_pipeline, has_loss_trace=self.has_loss_trace)
    
    def _buildable(self) -> bool:
        return self.train_recipe is not None or self.model_recipe is not None or self.save_location is not None
    
    def _create_eval_pipeline(self, tokenizer, encoder, generator):
        if self.eval_data is not None:
            eval_dataset = GraphDataset(self.eval_data, encoder.data_processor, tokenizer, self.model_recipe.graph_encoder_strategy, 
                                        self.train_recipe.is_classification)
            eval_dataloader = DataLoader(eval_dataset, 
                                         collate_fn=create_collate_fn(),
                                         batch_size=self.train_recipe.batch_size,
                                         num_workers=4, pin_memory=True)
            return EvalPipeline(eval_dataloader, single_process_accuracy, 1, tokenizer, encoder, generator, 
                                self.model_recipe.max_generation_len, device=self.device, data_name=self.eval_data.name)
        else:
            return None
    
    def _generate_error_msg(self) -> str:
        msg = ""
        if self.save_location is None:
            msg += "You have to add a save_location to build the training pipeline!\n"
        if self.model_recipe is None:
            msg += "You have to add a model recipe to build the training pipeline!\n"
        if self.train_recipe is None:
            msg += "You have to add a train recipe to build the training pipeline!\n"
        return msg
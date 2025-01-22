import torch

from pathlib import Path

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from GraphLanguageModel.pipelines.EvalPipeline import EvalPipeline
from GraphLanguageModel.pipelines.data_handling import GraphDataset, create_collate_fn
from GraphLanguageModel.pipelines.recipies import ModelRecipe, TrainRecipe
from GraphLanguageModel.pipelines.util import create_multiprocessed_accuracy, single_process_accuracy


class TrainPipeline:
    def __init__(self, dataloader: DataLoader, num_epochs: int, 
                 batch_size: int, early_stopping: int, optimizer,
                 tokenizer, encoder, generator, checkpointing_interval: int, 
                 save_location: Path, eval_pipeline: EvalPipeline = None, data_name: str = "") -> None:

        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.optimizer = optimizer
        
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.generator = generator

        self.checkpointing_interval = checkpointing_interval
        self.eval_pipeline = eval_pipeline
        self.data_name = data_name

        self.save_location = save_location
        self.encoder_save_location = self.save_location / "encoder"
        self.generator_save_location = self.save_location / "generator"
        self.progress_location = self.save_location / "progress"
        self.abort = 0

        self._read_progress()
        self.total = len(self.dataloader) - int(self.progress[1]/self.batch_size)
    
    def _read_progress(self):
        if self.progress_location.exists():
            with self.progress_location.open("r") as file:
                self.progress = eval(file.read())
        else:
            self.progress = [0, 0]

    def train(self):
        if self.eval_pipeline is not None:
            self.last_accuracy, _ = self.eval_run()
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
        batch_progress = int(self.progress[1]/self.batch_size)
        index = 0
        with tqdm(self.dataloader, postfix=f"Training {self.data_name}", 
                    total=len(self.dataloader) - batch_progress) as pbar:
            for (inputs, attention_mask), labels in pbar:
                with autocast(True, dtype=torch.bfloat16):
                    self.optimizer.zero_grad()
                    encoder_outputs = self.encoder(**inputs)
                    batch_loss = self.generator(encoder_outputs=encoder_outputs, labels=labels, attention_mask=attention_mask).loss
                    batch_loss.backward()
                    self.optimizer.step()
                    self.progress[1] += self.batch_size
                    index += 1
                    self.running_loss += batch_loss
                    pbar.set_description_str(f"Average loss last batch: {batch_loss}")
                    self._save_if_necessary(index)
                  
    def eval_run(self):
        return self.eval_pipeline.eval() if self.eval_pipeline is not None else (float("inf"), 0)    
    
    def _increment_early_stopping(self):
        if self.eval_pipeline is not None:
            avg_score, _ = self.eval_run()
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

    def _save_if_necessary(self, data_index: int):
        if (data_index + 1) % self.checkpointing_interval == 0 or data_index + 1 == self.total:
            if self.eval_pipeline is not None:
                current_accuracy, _= self.eval_run()
                if current_accuracy > self.last_accuracy:
                    self._save_model()
                    self.last_accuracy = current_accuracy
            else:
                self._save_model()
        self._save_progress()
    
    def _save_model(self):
        self.save_location.mkdir(exist_ok=True, parents=True)
        self.encoder.save_pretrained(self.encoder_save_location, from_pt=True)
        self.generator.save_pretrained(self.generator_save_location, from_pt=True)
        self.tokenizer.save_pretrained(self.encoder_save_location, from_pt=True)

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

    def build(self) -> TrainPipeline:
        if not self._buildable():
            raise ValueError(self._generate_error_msg())
        tokenizer, encoder, generator = self.model_recipe.build()
        optimizer = self.train_recipe.build(set(list(encoder.parameters()) + list(generator.parameters())))
        train_dataloader = DataLoader(self._create_dataset(encoder.data_processor, tokenizer), batch_size=self.train_recipe.batch_size,
                                collate_fn=create_collate_fn(torch.get_default_device(), encoder.data_processor, tokenizer, self.model_recipe.max_generation_len),
                                generator=torch.Generator(device=torch.get_default_device()),
                                shuffle=True)
        eval_pipeline = self._create_eval_pipeline(tokenizer, encoder, generator)
        return TrainPipeline(train_dataloader, self.train_recipe.num_epochs, self.train_recipe.batch_size, self.train_recipe.early_stopping, 
                            optimizer, tokenizer, encoder, generator, self.checkpointing_interval,
                            save_location=self.save_location, eval_pipeline=eval_pipeline, 
                            data_name=self.train_recipe.train_data.name)
    
    def _buildable(self) -> bool:
        return self.train_recipe is not None or self.model_recipe is not None or self.save_location is not None
    
    def _create_dataset(self, data_processor, tokenizer):
        return GraphDataset(self.train_recipe.train_data, data_processor, tokenizer, self.model_recipe.graph_encoder_strategy,
                            self.train_recipe.is_classification, neighborhood_size=self.train_recipe.neighborhood_size)
    
    def _create_eval_pipeline(self, tokenizer, encoder, generator):
        if self.eval_data is not None:
            eval_dataset = GraphDataset(self.eval_data, encoder.data_processor, tokenizer, self.model_recipe.graph_encoder_strategy, self.train_recipe.is_classification)
            eval_dataloader = DataLoader(eval_dataset, batch_size=self.train_recipe.batch_size*2,
                                    collate_fn=create_collate_fn(torch.get_default_device(), encoder.data_processor, tokenizer, self.model_recipe.max_generation_len))
            return EvalPipeline(eval_dataloader, single_process_accuracy, 1, tokenizer, encoder, generator, self.model_recipe.max_generation_len, data_name=self.eval_data.name)
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
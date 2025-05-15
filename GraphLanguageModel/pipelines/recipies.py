from pathlib import Path
from typing import Dict
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, AutoConfig

from GraphLanguageModel.pipelines.util import ModelCheckpoint
from utils.oop import str_to_optimizer

class TrainRecipe:
    def __init__(self, is_classification: bool, train_data: Path, num_epochs: int, batch_size: int, 
                 early_stopping: int, optimizer: str, optimizer_kwargs: Dict, learning_rate: float, 
                 neighborhood_size: int) -> None:
        self.is_classification = is_classification
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.learning_rate = learning_rate
        self.neighborhood_size = neighborhood_size

    def build(self, model_parameters):
        self.optimizer = self._create_optimizer(model_parameters)
        return self.optimizer        
    
    def _create_optimizer(self, parameters):
        optimizer_class = str_to_optimizer(self.optimizer)
        return optimizer_class(parameters, lr=self.learning_rate, **self.optimizer_kwargs)


class ModelRecipe:
    def __init__(self, encoder_modelcard: str, graph_encoder_strategy: str, 
                 generator_modelcard: str, model_checkpoint: ModelCheckpoint, 
                 max_generation_len: int = 30, gradient_checkpointing: bool = True):
        self.encoder = encoder_modelcard
        self.generator = generator_modelcard
        self.graph_encoder_strategy = graph_encoder_strategy
        self.max_generation_len = max_generation_len
        self.gradient_checkpointing = gradient_checkpointing
        self.model_checkpoint = model_checkpoint

    def build(self, device: str):
        tokenizer = self._load_tokenizer()
        encoder = self._load_encoder(device)
        generator = self._load_generator(device)
        generator.shared = encoder.shared
        return tokenizer, encoder, generator

    def _load_encoder(self, device: str):
        print(f"Load encoder from {self.encoder}")
        model = AutoModel.from_pretrained(self.encoder, trust_remote_code=True, device_map="auto", torch_dtype="bfloat16", revision='main')
        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model.to(device)

    def _load_generator(self, device: str):
        model_generation = None
        if self.generator is not None:
            print(f"Load generator from {self.generator}")
            model_generation = T5ForConditionalGeneration.from_pretrained(self.generator, device_map="auto", torch_dtype="bfloat16", trust_remote_code=True)
            if self.gradient_checkpointing:
                model_generation.gradient_checkpointing_enable()
            del model_generation.encoder  # we only need the decoder for generation. Deleting the encoder is optional, but saves memory.
        return model_generation.to(device)
    
    def _load_tokenizer(self):
        print(f"Load tokenizer from {self.encoder}")
        tokenizer = AutoTokenizer.from_pretrained(self.encoder, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        return tokenizer

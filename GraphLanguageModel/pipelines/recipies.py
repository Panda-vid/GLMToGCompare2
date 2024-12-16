from pathlib import Path
from typing import Dict
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration

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
    def __init__(self, encoder_modelcard: str, graph_encoder_strategy: str , generator_modelcard: str = None, max_generation_len: int = 512):
        self.encoder = encoder_modelcard
        self.generator = generator_modelcard
        self.graph_encoder_strategy = graph_encoder_strategy
        self.max_generation_len = max_generation_len

    def build(self):
        tokenizer = self._load_tokenizer()
        encoder = self._load_encoder()
        generator = self._load_generator()
        generator.shared = encoder.shared
        return tokenizer, encoder, generator

    def _load_encoder(self):
        print(f"Load encoder from {self.encoder}")
        model = AutoModel.from_pretrained(self.encoder, trust_remote_code=True, revision='main')
        return model

    def _load_generator(self):
        model_generation = None
        if self.generator is not None:
            print(f"Load generator from {self.generator}")
            model_generation = T5ForConditionalGeneration.from_pretrained(self.generator, trust_remote_code=True)
            del model_generation.encoder  # we only need the decoder for generation. Deleting the encoder is optional, but saves memory.
        return model_generation
    
    def _load_tokenizer(self):
        print(f"Load tokenizer from {self.encoder}")
        tokenizer = AutoTokenizer.from_pretrained(self.encoder, trust_remote_code=True)
        return tokenizer

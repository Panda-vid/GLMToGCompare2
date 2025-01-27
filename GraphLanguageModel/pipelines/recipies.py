from pathlib import Path
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration

from utils.oop import str_to_loss, str_to_optimizer

class TrainRecipe:
    def __init__(self, is_classification: bool, train_data: Path, num_epochs: int, batch_size: int, early_stopping: int, optimizer: str, learning_rate: float, loss: str, neighborhood_size: int) -> None:
        self.is_classification = is_classification
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.neighborhood_size = neighborhood_size

    def build(self, model_parameters):
        self.loss = self._create_loss()
        self.optimizer = self._create_optimizer(model_parameters)
        return self.loss, self.optimizer

    def _create_loss(self):
        loss_class = str_to_loss(self.loss)
        return loss_class(ignore_index=0)         
    
    def _create_optimizer(self, parameters):
        optimizer_class = str_to_optimizer(self.optimizer)
        return optimizer_class(parameters, lr=self.learning_rate)


class ModelRecipe:
    def __init__(self, encoder_modelcard: str, graph_encoder_strategy: str , generator_modelcard: str = None, max_generation_len: int = 30):
        self.encoder = encoder_modelcard
        self.generator = generator_modelcard
        self.graph_encoder_strategy = graph_encoder_strategy
        self.max_generation_len = max_generation_len
        self.gradient_checkpointing = True

    def build(self, device: str):
        tokenizer = self._load_tokenizer()
        encoder = self._load_encoder(device)
        generator = self._load_generator(device)
        generator.shared = encoder.shared
        return tokenizer, encoder, generator

    def _load_encoder(self, device: str):
        print(f"Load encoder from {self.encoder}")
        model = AutoModel.from_pretrained(self.encoder, trust_remote_code=True, revision='main', torch_dtype='auto', device_map="auto")
        if self.gradient_checkpointing:
                model.gradient_checkpointing_enable()
        return model.to(device)

    def _load_generator(self, device: str):
        model_generation = None
        if self.generator is not None:
            print(f"Load generator from {self.generator}")
            model_generation = T5ForConditionalGeneration.from_pretrained(self.generator, torch_dtype='auto', device_map="auto", trust_remote_code=True)
            if self.gradient_checkpointing:
                model_generation.gradient_checkpointing_enable()
            del model_generation.encoder  # we only need the decoder for generation. Deleting the encoder is optional, but saves memory.
        return model_generation.to(device)
    
    def _load_tokenizer(self):
        print(f"Load tokenizer from {self.encoder}")
        tokenizer = AutoTokenizer.from_pretrained(self.encoder, torch_dtype='auto', device_map="auto", trust_remote_code=True)
        return tokenizer

from argparse import Namespace
import json
from pathlib import Path
from random import randint, sample
from typing import Dict, Iterable, Tuple

from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, data_file: Path, data_processor, tokenizer, graph_encoder_strategy: str, is_classification: bool, 
                 neighborhood_size: int = 1000) -> None:
        self.is_classification = is_classification
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.graph_encoder_strategy = graph_encoder_strategy
        self.neighborhood_size = neighborhood_size
        self.data = self._load_data_into_memory(data_file)

    def _load_data_into_memory(self, data_file):
        with data_file.open("r") as file:
            return file.readlines()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Dict:
        instance = json.loads(self.data[index])
        return self._convert_data_instance(instance)

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
    
    def _convert_text_data(self, text_instance):
        return text_instance[randint(0, len(text_instance) - 1)] if isinstance(text_instance, list) else text_instance


def create_collate_fn(device, data_processor, tokenizer, max_generation_len):
    def collate_fn(batch: Iterable[Tuple[Namespace, str]]):
        inputs, labels = tuple(zip(*batch))
        return (
            data_processor.to_batch(data_instances=inputs, tokenizer=tokenizer, max_seq_len=None, device=device, return_attention_mask=True),
            tokenizer(labels, return_tensors="pt", padding=True).input_ids.to(device)
        )
    return collate_fn


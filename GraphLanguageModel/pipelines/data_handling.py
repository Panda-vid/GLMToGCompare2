from argparse import Namespace
import json
from pathlib import Path
from random import randint, sample
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset


class GraphDataset(Dataset):
    def __init__(self, data_file: Path, data_processor, tokenizer, graph_encoder_strategy: str, is_classification: bool, 
                 neighborhood_size: int = 1000, data_offset: int = 0) -> None:
        self.is_classification = is_classification
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.graph_encoder_strategy = graph_encoder_strategy
        self.neighborhood_size = neighborhood_size
        self.data_offset = data_offset
        self.data_file = data_file
        self.data = self._load_data()

    def _load_data(self):
        with self.data_file.open("r") as file:
            data = np.array(file.readlines(), dtype=np.dtypes.StringDType())
        return data[self.data_offset:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self._convert_data_instance(json.loads(self.data[index]))

    def _convert_data_instance(self, instance: Dict):
        if len(instance["graph"]) > self.neighborhood_size:
            graph = [instance["graph"][0]] + sample(instance["graph"][1:], self.neighborhood_size)
        else:
            graph = instance["graph"]
        label = "<extra_id_0> " + instance["output"][0] if self.is_classification else instance["output"]
        text = None
        if "text" in instance.keys():
            text = self._convert_text_data(instance["text"])
        return self.data_processor.encode_graph(tokenizer=self.tokenizer, g=graph, text=text, how=self.graph_encoder_strategy), label    
    
    def _convert_text_data(self, text_instance):
        return text_instance[randint(0, len(text_instance) - 1)] if isinstance(text_instance, list) else text_instance


def create_collate_fn():
    def collate_fn(batch: Iterable[Tuple[Namespace, str]]):
        inputs, labels = tuple(zip(*batch))
        return (inputs, labels)
    return collate_fn


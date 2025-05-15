import json
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from GraphLanguageModel.preprocessing.general import DataGenerator, Preprocessor


class DataGeneratorKBQADev(DataGenerator):
    def __init__(self, source_file: Path):
        self.source_file = source_file
        self._load_datapoints()
        super().__init__(self.generator, self.get_length)

    def _load_datapoints(self):
        with self.source_file.open("r") as file:
            self.data_points = json.load(file)

    def generator(self, source_file):   
        for i, data_point in tqdm(enumerate(self.data_points), desc=f"Preprocessing {self.source_file}", total=len(self.data_points)):
            yield i, data_point

    def get_length(self, source_file):
        return len(self.data_points)

def preprocess(data_point: Dict):
    processed = {}
    processed["graph"] = []
    processed["output"] = " ".join([f"<extra_id_{i}> {answer}" for i, answer in enumerate(data_point["answer"].values())])
    processed["text"] = data_point["question"]
    graph_entities = {v: k for k, v in data_point["qid_topic_entity"].items()}
    return processed, graph_entities

if __name__ == "__main__":
    data_path = Path("./data/").resolve()
    sources = [data_path/"qald-10-dev-en.json"]
    target_directory = data_path/"preprocessed"
    data_generator = DataGeneratorKBQADev(sources[0])
    preprocessor = Preprocessor(sources, target_directory, data_generator, preprocess)
    preprocessor.preprocess()

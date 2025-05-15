import json
from pathlib import Path
from typing import Callable, Dict, Generator, Iterable, List, Tuple

from GraphLanguageModel.preprocessing.online.wikidata.PropertyStore import PropertyStore
from GraphLanguageModel.preprocessing.online.wikidata.api_interaction import request_direct_neighborhoods_of_entities
from GraphLanguageModel.preprocessing.online.wikipedia.api_interaction import create_entity_to_wikidata_id


class DataGenerator:
    def __init__(self, generator: Generator[Tuple[int, Dict], None, None], get_length = Callable[[Path], int]) -> None:
        self.generator = generator
        self.get_length = get_length


class Preprocessor:
    def __init__(self, data_sources: Iterable[Path], target_directory: Path, data_point_generator: DataGenerator, data_point_preprocessing_func: Callable[[Dict], Tuple[Dict, str]], batch_size: int = 49) -> None:
        self.data_sources = data_sources
        self._verify_sources()
        self.target_directory = target_directory
        self._create_target_directory()
        self.property_store = PropertyStore(target_directory/"properties.jsonl")
        self.target_files = [self.target_directory / (data_file.stem + ".jsonl") for data_file in self.data_sources]
        self.batch_size = batch_size
        self._get_file_offsets()
        self.data_point_processing_func = data_point_preprocessing_func
        self.data_point_generator = data_point_generator
        
    def _verify_sources(self) -> None:
        for source in self.data_sources:
            if not source.exists():
                raise ValueError(f"No source found at '{source}'.")
            if not source.is_file():
                raise ValueError(f"Sources are required to be files. '{source}' does not point to a file.")
            if source.suffix not in [".jsonl", ".json"]:
                raise ValueError(f"Sources must be either .jsonl or .json files. Given {source.suffix} file.")

    def _create_target_directory(self) -> None:
        self.target_directory.mkdir(exist_ok=True, parents=True)

    def _get_file_offsets(self):
        self.file_offsets = []
        for target_file in self.target_files:
            if target_file.exists():
                with target_file.open('r') as file:
                    self.file_offsets.append(len(file.readlines()) + 1)
            else:
                self.file_offsets.append(0)

    def preprocess(self) -> None:
        for i in range(len(self.data_sources)):
            self._preprocess_file(i)

    def _preprocess_file(self, file_id: int) -> List[Dict]:
        contents = []
        batch = []
        source_file = self.data_sources[file_id]
        file_offset = self.file_offsets[file_id]
        target_file = self.target_files[file_id]
        source_length = self.data_point_generator.get_length(source_file)

        for i, data_point in self.data_point_generator.generator(source_file):
            if i + 1 > file_offset:
                batch.append(data_point)
                if len(batch) == self.batch_size or i >= source_length - 1:
                    contents += self._preprocess_batch(batch)
                    self._save_preprocessed(contents, target_file)
                    batch = []
                    contents = []

    def _preprocess_batch(self, batch: List[Dict]) -> Dict:
        processed_data = []
        graph_sources = []
        for data_point in batch:
            processed, graph_source = self._preprocess_data_point(data_point)
            processed_data.append(processed)
            graph_sources.append(graph_source)
            
        graph_contexts = self._get_neighborhood_graphs(graph_sources)
        for i, processed in enumerate(processed_data):
            processed["graph"] += graph_contexts[i]
        
        return processed_data

    def _save_preprocessed(self, contents: List[Dict], path: Path):
        contents = [json.dumps(entry) + "\n" for entry in contents]
        with path.open("a+") as file:
            file.writelines(contents)

    def _preprocess_data_point(self, data_point: Dict) -> Tuple[Dict, str]:
        return self.data_point_processing_func(data_point)
    
    def _get_neighborhood_graphs(self, source_entities: List[str]|List[Dict]) -> List[List[Tuple[str, str, str]]]:
        if type(source_entities[0]) is dict:
            anchor_entities = []
            neighborhoods = {}
            for neighborhood_sources in source_entities:
                anchor = list(neighborhood_sources.keys())[0]
                partial_neighborhoods = request_direct_neighborhoods_of_entities(list(neighborhood_sources.items()))
                neighborhoods[anchor] = [tup for neigh in partial_neighborhoods.values() for tup in neigh]
                anchor_entities.append(anchor)
            source_entities = anchor_entities
        else:
            label_to_id, source_entities = create_entity_to_wikidata_id(source_entities)
            neighborhoods = request_direct_neighborhoods_of_entities(list(label_to_id.items()))
        neighborhood_graphs = []
        for ent in source_entities:
            neighborhood_graphs.append(self._translate_relation_ids(neighborhoods[ent]) if ent is not None else [])
        return neighborhood_graphs

    def _translate_relation_ids(self, graph: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        return [(subj, self.property_store.get_property_label(rel), obj)for subj, rel, obj in graph]


def transform_input_to_GLM_triple(input: str) -> Tuple[str, str, str]:
    transformed = [s.strip() for s in input.split("[SEP]")]
    transformed.append("<extra_id_0>")
    assert len(transformed) == 3
    return tuple(transformed)

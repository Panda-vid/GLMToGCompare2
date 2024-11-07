import json

from pathlib import Path
from typing import Callable, Dict, List, Tuple

from tqdm import tqdm

from GraphLanguageModel.preprocessing.general import DataGenerator, Preprocessor, transform_input_to_GLM_triple


def generator(source_file: Path):
    with source_file.open("r") as file:
        for i, line in tqdm(enumerate(file), desc=f"Preprocessing {source_file}", total=get_filelength(source_file)):
            yield i, json.loads(line)


def create_preprocess(text_input_key: str = None) -> Callable[[Dict], Tuple[Dict, str]]:
    def preprocess(data_point: Dict) -> Tuple[Dict, str]:
        processed = {}
        processed["graph"] = [transform_input_to_GLM_triple(data_point["input"])]
        processed["output"] = transform_output(data_point["output"])
        if text_input_key is not None:
            processed["text"] = data_point["meta"][text_input_key]
        return processed, processed["graph"][0][0]
    return preprocess


def get_filelength(source_file: Path) -> int:
    with source_file.open("r") as file:
        filelength = len(file.readlines())
    return filelength


def transform_output(output: List[Dict]) -> List[str]:
    return [answer["answer"] for answer in output]


if __name__ == "__main__":
    data_path = Path("./test_data/").resolve()
    sources = [data_path/"structured_zeroshot-dev-kilt.jsonl"]
    target_directory = data_path / "preprocessed"
    text_input_key = "template_questions"
    preprocessor = Preprocessor(sources, target_directory, DataGenerator(generator, get_filelength), create_preprocess(text_input_key))
    preprocessor.preprocess()

    
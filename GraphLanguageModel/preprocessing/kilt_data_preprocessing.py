import json

from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Any


TEMPLATE_QUESTION_KEY = "template_questions"


def preprocess_files_in(directory: Path) -> None:
    processed_directory = directory / "preprocessed"
    processed_directory.mkdir(exist_ok=True, parents=True)
    for data_file in directory.glob("*.jsonl"):
        processed = preprocess_file(data_file)
        processed_file_path = processed_directory / data_file.name
        save_jsonl_file(processed_file_path, processed)


def save_jsonl_file(filepath: Path, contents: List[Dict]) -> None:
    assert filepath.suffix in [".jsonl"]
    print(f"Saving file {filepath}")
    contents = [json.dumps(entry) + "\n" for entry in contents]
    with filepath.open("w+") as file:
        file.writelines(contents)


def preprocess_file(data_file: Path) -> List[Dict]:
    contents = []
    with data_file.open("r") as file:
        for line in tqdm(file, desc=f"Preprocessing {data_file}"):
            entry = json.loads(line)
            contents.append(preprocess_line(entry, TEMPLATE_QUESTION_KEY in entry["meta"].keys()))
    return contents


def preprocess_line(entry: Dict, extract_template_questions: bool) -> Dict:
    processed_entry = {}
    processed_entry["input"] = transform_input_to_GLM_triple(entry["input"])
    processed_entry["output"] = transform_output(entry["output"])
    if extract_template_questions:
        processed_entry[TEMPLATE_QUESTION_KEY] = entry["meta"][TEMPLATE_QUESTION_KEY]
    return processed_entry


def transform_input_to_GLM_triple(input: str) -> Tuple[str, str, str]:
    transformed = [s.strip() for s in input.split("[SEP]")]
    transformed.insert(1, "<extra_id_0>")
    assert len(transformed) == 3
    return tuple(transformed)


def transform_output(output: List[Dict]) -> List[str]:
    return [transform_answer(answer) for answer in output]


def transform_answer(answer: Dict) -> Dict:
    return answer["answer"]


if __name__ == "__main__":
    data_path = Path("./GLMToGCompare2/data/").resolve()
    preprocess_files_in(data_path)
    
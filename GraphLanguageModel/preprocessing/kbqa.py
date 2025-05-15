import json
from pathlib import Path
import re
from typing import Dict, List, Tuple

from textblob import TextBlob, Word
from tqdm import tqdm

from GraphLanguageModel.preprocessing.general import DataGenerator, Preprocessor


def generator(source_file: Path):
    with source_file.open("r") as file:
        dataset_dict = json.load(file)
    for i, data_point in tqdm(enumerate(dataset_dict["questions"]), desc=f"Preprocessing {source_file}", total=len(dataset_dict["questions"])):
        yield i, data_point


def preprocess(data_point: Dict) -> Tuple[Dict, str]: 
    processed = {}
    processed["graph"] = []
    processed["output"] = data_point["query"]["sparql"]
    processed["text"], source_entity = extract_question_and_source_entity(data_point["question"])
    return processed, source_entity


def get_length(source_file: Path) -> int:
    with source_file.open("r") as file:
        dataset_dict = json.load(file)
    return len(dataset_dict["questions"])


def extract_question_and_source_entity(question_list: List[Dict]) -> Tuple[str, str]:
    for question in question_list:
        if question["language"] == "en":
            return question["string"], extract_source_entity(question)
    raise ValueError(f"English question Extraction failed.\nQuestion list:\n{question_list}")


def extract_source_entity(question: Dict) -> str:
    keywords = [keyword.strip() for keyword in question["keywords"].split(",")]

    possible_source_entities = [keyword for keyword in keywords if keyword[0].isupper()]
    if len(possible_source_entities) > 0:
        return possible_source_entities[0]

    for keyword in keywords:
        match = re.search(keyword, question["string"], re.IGNORECASE)
        if match is not None:
            matched_string = match.group()
            if matched_string[0].isupper():
                return matched_string
    
    blob = TextBlob(question["string"])

    noun_phrases = blob.noun_phrases
    for keyword in keywords:
        if keyword in noun_phrases:
            return keyword

    nouns = [noun for noun, tag in blob.tags if tag in ["NN", "NNS"]]
    nouns_sigularized = [Word(noun).singularize().string for noun in nouns]
    for keyword in keywords:
        if keyword in nouns or keyword in nouns_sigularized:
            return keyword
    
    return None


if __name__ == "__main__":
    data_path = Path("./data/").resolve()
    sources = [data_path/"qald-9-train-multilingual.json"]
    target_directory = data_path / "preprocessed"
    preprocessor = Preprocessor(sources, target_directory, DataGenerator(generator, get_length), preprocess)
    preprocessor.preprocess()
import json
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

def main():
    graph_input_path = Path("./qald-9-train-multilingual.jsonl").resolve()
    label_input_path = Path("../qald-9-train-multilingual.json").resolve()
    output_path = Path("./qald-9-train-en.jsonl").resolve()
    with label_input_path.open("r") as label_input_file:
        label_dataset = json.load(label_input_file)["questions"]

    with output_path.open("+a") as output_file:
        with graph_input_path.open("r") as graph_input_file:
            for i, line in enumerate(graph_input_file):
                transformed = {}
                entry = json.loads(line)
                transformed["graph"] = entry["graph"]
                transformed["text"] = entry["text"]
                transformed["output"] = get_output(label_dataset[i])
                if len(transformed["graph"]) > 0: 
                    output_file.write(json.dumps(transformed) + "\n")
            

def get_output(entry: Dict):
    answertype = entry["answertype"]
    if len(entry["answers"]) > 1:
        raise ValueError(f"Multiple answers detected at id {entry["id"]}.")
    if answertype == "boolean":
        answers = get_boolean_answer(entry)
    elif answertype in ["resource", "number", "string", "date", "uri"]:
        answers = get_resource_answers(entry)
    else:
        raise ValueError(f"Unknown answer type {answertype}.")
    return transform_to_t5_string(answers)

def get_boolean_answer(entry: Dict) -> List[str]:
    if entry["answers"][0]["boolean"]:
        return ["Yes"]
    else:
        return ["No"]
    
def get_resource_answers(entry: Dict) -> List[str]:
    value_type = entry["answers"][0]["head"]["vars"][0]
    answers = []
    for binding in entry["answers"][0]["results"]["bindings"]:
        answers.append(to_str(binding[value_type]["value"], value_type))
    return answers

def transform_to_t5_string(answers: List[str]) -> str:
    result = ""
    for i, answer in enumerate(answers):
        result += f"<extra_id_{i}> {answer} "
    return result.strip()

def to_str(value: str, value_type: str) -> str:
    if value_type == "uri":
        url_value = urlparse(value)
        result = Path(url_value.path).name.replace("_", " ")
    elif value_type in ["c", "string", "date"]:
        result = value
    return result

if __name__ == "__main__":
    main()


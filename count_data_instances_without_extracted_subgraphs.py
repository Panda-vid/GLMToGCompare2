import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(prog="count_data_instances_without_extracted_subgraphs")
parser.add_argument(
    "data_path",
    type=str,
)
args = parser.parse_args()


def main(parseargs):
    data_path = Path(parseargs.data_path)
    counter_neighborhood = 0
    total = 0
    with data_path.open("r") as f:
        for line in f.readlines():
            dt = json.loads(line)
            if len(dt["graph"]) > 0:
                counter_neighborhood += 1
            total += 1
    print(f"A neighborhood found in {counter_neighborhood} of {total} meaning {(counter_neighborhood/total) * 100}% of instances have a neighborhood.")


if __name__ == "__main__":
   main(args) 
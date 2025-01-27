import argparse
from pathlib import Path
from typing import List


class KeywordAction(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: str, option_string: str=None) -> None:
        keyword_dict = {}
        for arg in values:
            pieces = arg.split("=")

            if len(pieces) == 2:
                keyword_dict[pieces[0]] = pieces[1]
            else:
                raise argparse.ArgumentError("Keyword arguments must be passed in the following form: 'keyword=value'.")
        
        setattr(namespace, self.dest, keyword_dict)


class PathAction(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: str, option_string: str=None) -> None:
        setattr(namespace, self.dest, Path(values).resolve())


def problem_type_to_classification_bool(problem_type: str):
    return problem_type == "classification" 
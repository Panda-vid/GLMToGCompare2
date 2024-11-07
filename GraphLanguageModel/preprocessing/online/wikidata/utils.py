from re import match
from urllib.parse import urlparse
from pathlib import Path


def is_entity_id(inp: str) -> bool:
    return is_id(inp, "Q")


def is_property_id(inp: str) -> bool:
    return is_id(inp, "P")


def is_id(inp: str, prefix: str) -> bool:
    return match(f"{prefix}[0-9]+", inp) is not None


def url_to_id(wikidata_url: str) -> str:
    assert contains_wikidata_link(wikidata_url), f"The given url string is not from wikidata! Given: '{wikidata_url}'."
    return Path(urlparse(wikidata_url).path).name


def contains_wikidata_link(value: str) -> bool:
    return "http://www.wikidata.org" in value
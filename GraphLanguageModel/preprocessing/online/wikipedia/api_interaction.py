import json
import re

import requests

from typing import List, Dict

from GraphLanguageModel.preprocessing.online.rate_limiting import RateManager


def create_entity_to_wikidata_id(entities: List[str]) -> Dict:
    response_content = request_pageproperties(entities)
    ent_to_wiki_id = {}
    for ent in entities:
        if ent is not None:
            ent_to_wiki_id[ent] = None
    
    for _, ent in response_content["query"]["pages"].items():
        ent_label = ent["title"]
        if ent_label not in ent_to_wiki_id:
            ent_to_wiki_id, entities = normalize_label(ent_label, entities, ent_to_wiki_id)
        assert ent_label in ent_to_wiki_id, f"Cant create entity label to wikidata id translation dict because '{ent_label}' is not in {list(ent_to_wiki_id)}."
        if "pageprops" in ent.keys():
            if "wikibase_item" in ent["pageprops"].keys():
                ent_to_wiki_id[ent_label] = ent["pageprops"]["wikibase_item"]

    return ent_to_wiki_id, entities


@RateManager.limit_rate("wikipedia", 2)
def request_pageproperties(entities: List[str]):
    url = "https://en.wikipedia.org/w/api.php?"
    params = {"action": "query", "prop": "pageprops", "titles": "|".join([ent for ent in entities if ent is not None]), "format": "json"}
    
    response = requests.get(url, params=params)

    if 200 <= response.status_code < 500:
        response.raise_for_status()

    return json.loads(response.content)


def normalize_label(label: str, entities: List[str], ent_to_wiki_id: Dict):
    lower_mask = [ent.lower() == label.lower() if ent is not None else False for ent in entities]
    if True in lower_mask:
        ent_to_wiki_id, entities = correct_by_mask(label, lower_mask, entities, ent_to_wiki_id)
        return ent_to_wiki_id, entities
    
    space_mask = [" ".join(ent.split()) == label if ent is not None else False for ent in entities]
    if True in space_mask:
        ent_to_wiki_id, entities = correct_by_mask(label, space_mask, entities, ent_to_wiki_id)
        return ent_to_wiki_id, entities
    
    match_mask = [re.match(label, ent) is not None if ent is not None else False for ent in entities]
    if True in match_mask:
        ent_to_wiki_id, entities = correct_by_mask(label, match_mask, entities, ent_to_wiki_id)
        return ent_to_wiki_id, entities
    
    underscore_mask = [" ".join(ent.split("_")) == label if ent is not None else False for ent in entities]
    if True in underscore_mask:
        ent_to_wiki_id, entities = correct_by_mask(label, underscore_mask, entities, ent_to_wiki_id)
        return ent_to_wiki_id, entities 
    
    len_overlaps = [len(set(label.lower()).intersection(set(ent.lower()))) if ent is not None else False for ent in entities]
    sorted_len_overlaps = sorted(len_overlaps)
    overlap_mask = [sorted_len_overlaps[0] == overlap_len for overlap_len in len_overlaps]
    return correct_by_mask(label, overlap_mask, entities, ent_to_wiki_id)


def correct_by_mask(label: str, ent_mask: List[bool], entities: List[str], ent_to_wiki_id: Dict):
    for i, overlap in enumerate(ent_mask):
        if overlap:
            wrong_ent_label = entities[i]
            entities[i] = label
            ent_to_wiki_id.pop(wrong_ent_label, None)
            ent_to_wiki_id[label] = None
    return ent_to_wiki_id, entities


if __name__ == "__main__":
    print(create_entity_to_wikidata_id(["Douglas Adams", "USA-88"]))
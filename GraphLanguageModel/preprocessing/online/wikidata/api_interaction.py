import time

from typing import List, Dict, Tuple

from qwikidata.sparql import return_sparql_query_results
from qwikidata.entity import WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api
from requests.exceptions import JSONDecodeError

from GraphLanguageModel.preprocessing.online.rate_limiting import RateManager
from GraphLanguageModel.preprocessing.online.wikidata.utils import is_entity_id, is_property_id, url_to_id


@RateManager.limit_rate("wikidata", 1)
def request_property_label(property_id: str):
    assert is_property_id(property_id), f"Invalid property ID given! ID should have the form P<integer>, e.g. P31 = instance of. You gave {property_id}."
    return WikidataProperty(get_entity_dict_from_api(property_id)).get_label()


@RateManager.limit_rate("wikidata", 1)
def request_direct_neighborhoods_of_entities(sources: List[Tuple[str, str]]) -> Dict:
    source_ids = []
    source_id_to_label ={}
    neighborhoods = {}
    for source_label, source_id in sources:
        neighborhoods[source_label] = []
        if source_id is not None:
            assert is_entity_id(source_id), f"Invalid entity ID given! ID should have the for Q<integer>, e.g. Q42 = Douglas Adams. You gave {source_id}."
            source_ids.append(f"(wd:{source_id})")
            source_id_to_label[source_id] = source_label

    sparql_query = f"""
    SELECT ?source ?relationLabel ?entity ?entityLabel WHERE {{
                         # source relation
        VALUES (?source) {{{" ".join(source_ids)}}}

        # find entity labels in relation with source
        ?source ?relation ?entity.

        # only show direct relations
        [] wikibase:directClaim ?relation.

        # get label service
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en".}}
    }}
    """
    try:
        result = return_sparql_query_results(sparql_query)
    except JSONDecodeError:
        print("Retrying to retrieve neighborhood")
        time.sleep(2)
        return request_direct_neighborhoods_of_entities(sources)
    
    for result_dict in (result["results"]["bindings"]):
        source_id = url_to_id(result_dict["source"]["value"])
        source_label = source_id_to_label[source_id]
        relation_label = url_to_id(result_dict["relationLabel"]["value"])
        entity_label = result_dict["entityLabel"]["value"]
        neighborhoods[source_label].append((source_label, relation_label, entity_label))
    return neighborhoods


if __name__ == "__main__":
    print(request_direct_neighborhoods_of_entities([("Q42", "Douglas Adams")]))
import json

from pathlib import Path
from typing import List

from utils.oop import Singleton
from .utils import is_property_id
from .api_interaction import request_property_label


class Property:
    def __init__(self, pid: str, label: str = None):
        self.pid = pid
        self.label = label
        if self.label is None:
            self.label = request_property_label(self.pid)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)


class PropertyStore(Singleton):
    def init(self, store_location: Path = Path("../../../data/wikidata_properties.json")):
        self._store_location = store_location.resolve()
        self._initialize_properties()

    def _initialize_properties(self) -> None:
        self._properties = {}
        if self._store_location.exists():
            self._load_store()
        else:
            self._create_store()

    def _load_store(self) -> None:
        with self._store_location.open("r") as store:
            for line in store:
                prop = Property(**json.loads(line))
                self._add_property_to_dict(prop)
    
    def _create_store(self) -> None:
        self._store_location.parent.mkdir(parents=True, exist_ok=True)
        self._store_location.open("a").close()

    def get_property_label(self, property_id: str) -> str:
        if not self._is_saved(property_id):
            self._add_property(property_id)
        return self._properties[property_id]

    def _add_property(self, pid: str) -> None:
        assert is_property_id(pid), f"Invalid property ID given! ID should have the form P<integer>, e.g. P31 = instance of. You gave {pid}."
        prop = Property(pid)
        self._add_property_to_dict(prop)
        self._append_store(prop)

    def _add_property_to_dict(self, prop: Property):
        self._properties[prop.pid] = prop.label

    def _is_saved(self, pid: str) -> bool:
        return pid in self._properties
    
    def _append_store(self, prop: Property) -> None:
        with self._store_location.open('a') as store:
            line = prop.to_json() + "\n"
            store.write(line)


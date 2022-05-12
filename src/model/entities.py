import logging
from typing import Dict

class Profile:

    def __init__(self, url):
        self.attributes = set()
        if url != None:
            self.url = url
        else:
            logging.error("url cannot be null")

    def add_attribute(self, property_name, property_value):
        self.attributes.add(Attribute(property_name, property_value))


class Attribute:

    def __init__(self, name, value) -> any:
        self.value = value
        self.name = name

    def __eq__(self, __o: object) -> bool:
        if __o is None or not isinstance(__o, Attribute) or __o.name != self.name:
            return False
        return True

class AttributeClusters:

    def __init__(self, entropy, mapping):
        self.clusters_entropy = entropy
        self.attribute_to_cluster_id: Dict[str, int] = mapping

    def get_num_of_clusters(self) -> int:
        return len(self.clusters_entropy)

    def get_clusters_entropy(self, cluster_id) -> float:
        return self.clusters_entropy[cluster_id]

    def get_cluster_id(self, attribute_name: str) -> int:
        return self.attribute_to_cluster_id.get(attribute_name)


class AbstractBlock:

    _block_index = -1
    _comparisons = 0
    _entropy = 1.0
    _utility_measure = -1

    def __init__(self):
        pass
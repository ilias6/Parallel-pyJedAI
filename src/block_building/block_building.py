import logging
import os
import sys
from data_model.entities import AttributeClusters
from typing import List


info = logging.info
error = logging.error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utilities.tokenizer import Tokenizer


class AbstractBlockBuilding:


    num_of_entities_d1: int
    num_of_entities_d2: int

    blocks: list

    entity_profiles_d1: list
    entity_profiles_d2: list

    inverted_index_d1: dict
    inverted_index_d2: dict

    schema_clusters: List[AttributeClusters]

    def __init__(self):
        self.is_using_cross_entropy: bool = False

    def build_blocks(self):
        if self.schema_clusters:
            self.index_entities(self.inverted_index_d1, self.entity_profiles_d1, self.schema_clusters[0])
        else:
            self.index_entities(self.inverted_index_d1, self.entity_profiles_d1)

        if self.inverted_index_d2:
            if self.schema_clusters:
                self.index_entities(self.inverted_index_d1, self.entity_profiles_d1, self.schema_clusters[1])
            else:
                self.index_entities(self.inverted_index_d1, self.entity_profiles_d1)

    def get_blocks(self, profiles_d1, profiles_d2 = None, schema_clusters = None):

        info("Applying -- with the following configuration : --")

        if profiles_d1 == None:
            logging.error("First list of entity profiles is null! The first argument should always contain entities.")
            return None
        
        self.entity_profiles_d1 = profiles_d1
        self.num_of_entities_d2 = len(profiles_d1)

        if profiles_d2:
            self.entity_profiles_d2 = profiles_d2
            self.num_of_entities_d2 = len(profiles_d2)

        self.schema_clusters = schema_clusters


    def index_entities(self, index, entities, schema_clusters=None) -> None:
        counter = 0
        for profile in entities:
            all_keys = set()
            for attr in profile.attributes:
                if schema_clusters:
                    cluster_id = schema_clusters.get_cluster_id(attr.name)
                
                
                


                



    def __str__(self) -> str:
        pass


class StandardBlocking(AbstractBlockBuilding):

    _method_name = "Standard Blocking"
    _method_info = _method_name + ": it creates one block for every token in the attribute values of at least two entities."

    def __init__(self) -> any:
        super().__init__()


    def get_blocking_keys(self, attribute_value) -> list:
        tok = Tokenizer(ngrams=1, is_char_tokenization=False, return_type='list')
        return tok.process(attribute_value)

    

    

    
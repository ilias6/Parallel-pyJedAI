from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from nltk import ngrams
from nltk.tokenize import word_tokenize
from pyjedai.datamodel import Block, Data
from typing import List, Tuple
import random
from queue import PriorityQueue
import math
import sys
from time import time
from networkx import Graph
import inspect

# ----------------------- #
# Constants
# ----------------------- #
EMPTY = -1

# ----------------------- #
# Utility Methods
# ----------------------- #
def create_entity_index(blocks: dict, is_dirty_er: bool) -> dict:
    """Creates a dict of entity ids to block keys
        Example:
            e_id -> ['block_key_1', ..]
            ...  -> [ ... ]
    """
    entity_index = {}
    for key, block in blocks.items():
        for entity_id in block.entities_D1:
            entity_index.setdefault(entity_id, set())
            entity_index[entity_id].add(key)

        if not is_dirty_er:
            for entity_id in block.entities_D2:
                entity_index.setdefault(entity_id, set())
                entity_index[entity_id].add(key)

    return entity_index

def are_matching(entity_index, id1, id2) -> bool:
    '''
    id1 and id2 consist a matching pair if:
    - Blocks: intersection > 0 (comparison of sets)
    - Clusters: cluster-id-j == cluster-id-i (comparison of integers)
    '''

    if len(entity_index) < 1:
        raise ValueError("No entities found in the provided index")
    if isinstance(list(entity_index.values())[0], set): # Blocks case
        return len(entity_index[id1].intersection(entity_index[id2])) > 0
    return entity_index[id1] == entity_index[id2] # Clusters case

def drop_big_blocks_by_size(blocks: dict, max_block_size: int, is_dirty_er: bool) -> dict:
    """Drops blocks if:
        - Contain only one entity
        - Have blocks with size greater than max_block_size

    Args:
        blocks (dict): Blocks.
        max_block_size (int): Max block size. If is greater that this, block will be rejected.
        is_dirty_er (bool): Type of ER.

    Returns:
        dict: New blocks.
    """
    return dict(filter(
        lambda e: not block_with_one_entity(e[1], is_dirty_er)
                    and e[1].get_size() <= max_block_size,
                blocks.items()
        )
    )

def drop_single_entity_blocks(blocks: dict, is_dirty_er: bool) -> dict:
    """Removes one-size blocks for DER and empty for CCER
    """
    return dict(filter(lambda e: not block_with_one_entity(e[1], is_dirty_er), blocks.items()))

def block_with_one_entity(block: Block, is_dirty_er: bool) -> bool:
    """Checks for one entity blocks.

    Args:
        block (Block): Block of entities.
        is_dirty_er (bool): Type of ER.

    Returns:
        bool: True if it contains only one entity. False otherwise.
    """
    return True if ((is_dirty_er and len(block.entities_D1) == 1) or \
        (not is_dirty_er and (len(block.entities_D1) == 0 or len(block.entities_D2) == 0))) \
            else False

def print_blocks(blocks: dict, is_dirty_er: bool) -> None:
    """Prints all the contents of the block index.

    Args:
        blocks (_type_):  Block of entities.
        is_dirty_er (bool): Type of ER.
    """
    print("Number of blocks: ", len(blocks))
    for key, block in blocks.items():
        block.verbose(key, is_dirty_er)

def print_candidate_pairs(blocks: dict) -> None:
    """Prints candidate pairs index in natural language.

    Args:
        blocks (dict): Candidate pairs structure.
    """
    print("Number of blocks: ", len(blocks))
    for entity_id, candidates in blocks.items():
        print("\nEntity id ", "\033[1;32m"+str(entity_id)+"\033[0m", " is candidate with: ")
        print("- Number of candidates: " + "[\033[1;34m" + \
            str(len(candidates)) + " entities\033[0m]")
        print(candidates)

def print_clusters(clusters: list) -> None:
    """Prints clusters contents.

    Args:
        clusters (list): clusters.
    """
    print("Number of clusters: ", len(clusters))
    for (cluster_id, entity_ids) in zip(range(0, len(clusters)), clusters):
        print("\nCluster ", "\033[1;32m" + \
              str(cluster_id)+"\033[0m", " contains: " + "[\033[1;34m" + \
            str(len(entity_ids)) + " entities\033[0m]")
        print(entity_ids)

def text_cleaning_method(col):
    """Lower clean.
    """
    return col.str.lower()

def chi_square(in_array: np.array) -> float:
    """Chi Square Method

    Args:
        in_array (np.array): Input array

    Returns:
        float: Statistic computation of Chi Square.
    """
    row_sum, column_sum, total = \
        np.sum(in_array, axis=1), np.sum(in_array, axis=0), np.sum(in_array)
    sum_sq = expected = 0.0
    for r in range(0, in_array.shape[0]):
        for c in range(0, in_array.shape[1]):
            expected = (row_sum[r]*column_sum[c])/total
            sum_sq += ((in_array[r][c]-expected)**2)/expected
    return sum_sq


def batch_pairs(iterable, batch_size: int = 1):
    """
    Generator function that breaks an iterable into batches of a set size.
    :param iterable: The iterable to be batched.
    :param batch_size: The size of each batch.
    """
    return (iterable[pos:pos + batch_size] for pos in range(0, len(iterable), batch_size))

def get_sorted_blocks_shuffled_entities(dirty_er: bool, blocks: dict) -> List[int]:
    """Sorts blocks in alphabetical order based on their token, shuffles the entities of each block, concatenates the result in a list

    Args:
        blocks (Dict[Block]): Dictionary of type token -> Block Instance

    Returns:
        List[Int]: List of shuffled entities of alphabetically, token sorted blocks
    """
    sorted_entities = []
    for _, block in sorted(blocks.items()):
        _shuffled_neighbors = list(block.entities_D1 | block.entities_D2 if not dirty_er else block.entities_D1)
        random.shuffle(_shuffled_neighbors)
        sorted_entities += _shuffled_neighbors

    return sorted_entities

class Tokenizer(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def tokenize(self, text: str) -> list:
        pass

class WordQgrammsTokenizer(Tokenizer):
    
    def __init__(self, q: int = 3) -> None:
        super().__init__()
        self.q = q
    
    def tokenize(self, text: str) -> list:
        return [' '.join(gram) for gram in list(ngrams(word_tokenize(text), self.q))]


class Tokenizer(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def tokenize(self, text: str) -> list:
        pass

class WordQgrammsTokenizer(Tokenizer):
    
    def __init__(self, q: int = 3) -> None:
        super().__init__()
        self.q = q
    
    def tokenize(self, text: str) -> list:
        return [' '.join(gram) for gram in list(ngrams(word_tokenize(text), self.q))]

class SubsetIndexer(ABC):
    """Stores the indices of retained entities of the initial datasets,
       calculates and stores the mapping of element indices from new to old dataset (id in subset -> id in original)
    """
    def __init__(self, blocks: dict, data: Data, subset : bool):
        self.d1_retained_ids: list[int] = None
        self.d2_retained_ids : list[int] = None
        self.subset : bool = subset
        self.store_retained_ids(blocks, data)

    def from_source_dataset(self, entity_id : int, data: Data) -> bool:
        return entity_id < data.dataset_limit

    def store_retained_ids(self, blocks: dict, data: Data) -> None:
        """Stores lists contains the ids of entities that we retained from both datasets
           in ascending order
        Args:
            blocks (dict): Mapping from entity id to a set of its neighbors ids
            data (Data): Dataset Module
        """

        if(not self.subset):
            self.d1_retained_ids = list(range(data.num_of_entities_1))

            if(not data.is_dirty_er):
                self.d2_retained_ids = list(range(data.num_of_entities_1, data.num_of_entities_1 + data.num_of_entities_2))
        else:
            _d1_retained_ids_set: set[int] = set()
            _d2_retained_ids_set: set[int] = set()

            if(data.is_dirty_er):
                _d1_retained_ids_set = set(blocks.keys())
                for neighbors in blocks.values():
                    _d1_retained_ids_set = _d1_retained_ids_set.union(neighbors)
                self.d1_retained_ids = sorted(list(_d1_retained_ids_set))
                self.d2_retained_ids = []
            else:

                for entity in blocks.keys():
                    if(self.from_source_dataset(entity, data)): _d1_retained_ids_set.add(entity) 
                    else: _d2_retained_ids_set.add(entity)

                    neighbors = blocks[entity]
                    for neighbor in neighbors:
                        if(self.from_source_dataset(neighbor, data)): _d1_retained_ids_set.add(entity)
                        else: _d2_retained_ids_set.add(entity)

                self.d1_retained_ids = sorted(list(_d1_retained_ids_set))
                self.d2_retained_ids = sorted(list(_d2_retained_ids_set))
                
class PositionIndex(ABC):
    """For each entity identifier stores a list of index it appears in, within the list of shuffled entities of sorted blocks

    Args:
        ABC (ABC): ABC Module 
    """
    
    def __init__(self, num_of_entities: int, sorted_entities: List[int]) -> None:
        self._num_of_entities = num_of_entities
        self._counters = self._num_of_entities * [0]
        self._entity_positions = [[] for _ in range(self._num_of_entities)]
        
        for entity in sorted_entities:
            self._counters[entity]+=1
            
        for i in range(self._num_of_entities):
            self._entity_positions[i] = [0] * self._counters[i]
            self._counters[i] = 0
            
        for index, entity in enumerate(sorted_entities):
            self._entity_positions[entity][self._counters[entity]] = index
            self._counters[entity] += 1
            
    def get_positions(self, entity: int):
        return self._entity_positions[entity]

class EntityScheduler(ABC):
    """Stores information about the neighborhood of a given entity ID:
    - ID : The identifier of the entity as it is defined within the original dataframe
    - Total Weight : The total weight of entity's neighbors
    - Number of Neighbors : The total number of Neighbors
    - Neighbors : Entity's neighbors sorted in descending order of weight
    - Stage : Insert / Pop stage (entities stored in ascending / descending weight order)

    Args:
        ABC (ABC): ABC Module 
    """
    
    def __init__(self, id : int) -> None:
        self._id : int = id
        self._neighbors : PriorityQueue = PriorityQueue()
        self._neighbors_num : int = 0
        self._total_weight : float = 0.0
        self._average_weight : float = None
        
    def _insert(self, neighbor_id: int, weight : float) -> None:
        self._neighbors.put((-weight, neighbor_id))
        self._update_neighbors_counter_by(1)
        self._update_total_weight_by(weight)
            
    def _pop(self) -> Tuple[float, int]:
        if(self._empty()):
            raise ValueError("No neighbors to pop!")
        
        _weight, _neighbor_id = self._neighbors.get()
        self._update_neighbors_counter_by(-1)
        self._update_total_weight_by(_weight)
        
        return -_weight, _neighbor_id
    
    def _empty(self) -> bool:
        return self._neighbors.empty()
        
    def _update_total_weight_by(self, weight) -> None:
        self._total_weight = self._total_weight + weight
        
    def _update_neighbors_counter_by(self, count) -> None:
        self._neighbors_num = self._neighbors_num + count
        
    def _get_neighbors_num(self) -> int:
        return self._neighbors_num
    
    def _get_total_weight(self) -> float:
        return self._total_weight
    
    def _get_average_weight(self) -> float:
        if(self._average_weight is None):
            self._average_weight = 0.0 if not self._get_neighbors_num() else (float(self._get_total_weight()) / float(self._get_neighbors_num()))
            return self._average_weight
        else:
            return self._average_weight
    
    def __eq__(self, other):
        if isinstance(other, EntityScheduler):
            return self._get_average_weight() == other._get_average_weight()
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, EntityScheduler):
            return self._get_average_weight() < other._get_average_weight()
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, EntityScheduler):
            return self._get_average_weight() > other._get_average_weight()
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, EntityScheduler):
            return self._get_average_weight() <= other._get_average_weight()
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, EntityScheduler):
            return self._get_average_weight() >= other._get_average_weight()
        return NotImplemented 
    
class DatasetScheduler(ABC):
    """Stores a dictionary [Entity -> Entity's Neighborhood Data (Whoosh Neighborhood)]
       Supplies auxiliarry functions for information retrieval from the sorted dataset

    Args:
        ABC (ABC): ABC Module
    """
     
    def __init__(self, budget : float = float('inf'), entity_ids : List[int] = [], global_top : bool = False) -> None:
        self._budget : float = budget
        self._total_entities : int = len(entity_ids)
        self._neighborhoods : dict = {}
        # global emission case
        self._global_top : bool = global_top
        self._all_candidates = PriorityQueue() if self._global_top else None
        for entity_id in entity_ids:  
            self._neighborhoods[entity_id] = EntityScheduler(id=entity_id)
        # used in defining proper emission strategy
        self._sorted_entities : List[int] = None
        self._current_neighborhood_index : int = 0
        self._current_entity : int = None
        self._current_neighborhood : EntityScheduler = None
            
    def _insert_entity_neighbor(self, entity : int, neighbor : int, weight : float) -> None:
        if(not self._global_top):
            _neighborhood = EntityScheduler(entity) if entity in self._neighborhoods else self._neighborhoods[entity]
            _neighborhood._insert(neighbor, weight)
        else:
            self._all_candidates.put((-weight, entity, neighbor))
        
    def _pop_entity_neighbor(self, entity : int) -> Tuple[float, int]:
        return self._neighborhoods[entity]._pop()
    
    def _get_entity_neighborhood(self, entity : int) -> EntityScheduler:
        return self._neighborhoods[entity]
    
    def _entity_has_neighbors(self, entity : int) -> bool:
        return not self._neighborhoods[entity]._empty()
    
    def _sort_neighborhoods_by_avg_weight(self) -> None:
        """Store a list of entity ids sorted in descending order of the average weight of their corresponding neighborhood"""
        self._sorted_entities : List = sorted(self._neighborhoods, key=lambda entity: self._neighborhoods[entity]._get_average_weight(), reverse=True)
    
    def _get_current_neighborhood(self) -> EntityScheduler:
        return self._neighborhoods[self._current_entity]
        
    def _enter_next_neighborhood(self) -> None:
        """Sets the next in descending average weight order neighborhood
        """
        _curr_nei_idx : int = self._current_neighborhood_index
        self._current_neighborhood_index = _curr_nei_idx + 1 if _curr_nei_idx + 1 < self._total_entities else 0
        self._current_entity = self._sorted_entities[self._current_neighborhood_index]
        self._current_neighborhood = self._neighborhoods[self._current_entity]
        
    def _successful_emission(self, pair : Tuple[float, int, int]) -> bool:
        _score, _entity, _neighbor = pair
                
        if(self._emitted_comparisons < self._budget):
            self._emitted_pairs.append((_score, _entity, _neighbor))
            self._checked_entities.add(canonical_swap(_entity, _neighbor))
            self._emitted_comparisons += 1
            return True
        else:
            return False
        
    def _emit_pairs(self, method : str) -> List[Tuple[float, int, int]]:
        """Emits candidate pairs according to specified method

        Args:
            method (str): Emission Method
            data (Data): Dataset Module

        Returns:
            List[Tuple[int, int]]: List of candidate pairs
        """
        
        self._method : str = method
        self._emitted_pairs = []
        self._emitted_comparisons = 0    
        self._checked_entities = set()
        
        if(self._method == 'Global'):
            while(not self._all_candidates.empty()):
                score, sorted_entity, neighbor = self._all_candidates.get()
                if(canonical_swap(sorted_entity, neighbor) not in self._checked_entities):
                    if(not self._successful_emission(pair=(-score, sorted_entity, neighbor))):
                        return self._emitted_pairs
                
                return self._emitted_pairs
            
        
        if(self._method == 'HB'):
            for sorted_entity in self._sorted_entities:
                if(self._entity_has_neighbors(sorted_entity)):
                    score, neighbor = self._pop_entity_neighbor(sorted_entity)
                    if(canonical_swap(sorted_entity, neighbor) not in self._checked_entities):
                        if(not self._successful_emission(pair=(score, sorted_entity, neighbor))):
                            return self._emitted_pairs
                   
        if(self._method == 'HB' or self._method == 'DFS'):            
            for sorted_entity in self._sorted_entities:
                while(self._entity_has_neighbors(sorted_entity)):
                    score, neighbor = self._pop_entity_neighbor(sorted_entity)
                    if(canonical_swap(sorted_entity, neighbor) not in self._checked_entities):
                        if(not self._successful_emission(pair=(score, sorted_entity, neighbor))):
                            return self._emitted_pairs
        else:
            _emissions_left = True
            while(_emissions_left):
                _emissions_left = False
                for sorted_entity in self._sorted_entities:
                    if(self._entity_has_neighbors(sorted_entity)):
                        score, neighbor = self._pop_entity_neighbor(sorted_entity)
                        if(canonical_swap(sorted_entity, neighbor) not in self._checked_entities):
                            if(not self._successful_emission(pair=(score, sorted_entity, neighbor))):
                                return self._emitted_pairs
                            _emissions_left = True
        return self._emitted_pairs
    
class PredictionData(ABC):
    """Auxiliarry module used to store basic information about the to-emit, predicted pairs
       It is used to retrieve that data efficiently during the evaluation phase, and subsequent storage of emission data (e.x. total emissions)

    Args:
        ABC (ABC): ABC Module
    """
    def __init__(self, name : str, predictions, tps_checked = dict) -> None:
        self.set_name(name)
        self.set_tps_checked(tps_checked)
        self.set_predictions(self._format_predictions(predictions))
        # Pairs have not been emitted yet - Data Module has not been populated with performance data
        self.set_total_emissions(None)
        self.set_normalized_auc(None)
        self.set_cumulative_recall(None)
    
    def _format_predictions(self, predictions) -> List[Tuple[int, int]]:
        """Transforms given predictions into a list of duplets (candidate pairs)
           Currently Graph and Default input is supported

        Args:
            predictions (Graph / List[Tuple[int, int]]): Progressive Matcher predictions

        Returns:
            List[Tuple[int, int]]: Formatted Predictions
        """
        return [edge[:3] for edge in predictions.edges] if isinstance(predictions, Graph) else predictions
        
    def get_name(self) -> str:
        return self._name
    
    def get_predictions(self) -> List[Tuple[float, int, int]]:
        return self._predictions
    
    def get_tps_checked(self) -> dict:
        return self._tps_checked
    
    def get_total_emissions(self) -> int:
        if(self._total_emissions is None): raise ValueError("Pairs not emitted yet - Total Emissions are undefined")
        return self._total_emissions
    
    def get_normalized_auc(self) -> float:
        if(self._normalized_auc is None): raise ValueError("Pairs not emitted yet - Normalized AUC is undefined")
        return self._normalized_auc
    
    def get_cumulative_recall(self) -> float:
        if(self._cumulative_recall is None): raise ValueError("Pairs not emitted yet - Cumulative Recall is undefined")
        return self._cumulative_recall
    
    def set_name(self, name : str):
        self._name : str = name
    
    def set_predictions(self, predictions : List[Tuple[float, int, int]]) -> None:
        self._predictions : List[Tuple[float, int, int]] = predictions
    
    def set_tps_checked(self, tps_checked : dict) -> None:
        self._tps_checked : dict = tps_checked
    
    def set_total_emissions(self, total_emissions : int) -> None:
        self._total_emissions : int = total_emissions
        
    def set_normalized_auc(self, normalized_auc : float) -> None:
        self._normalized_auc : float = normalized_auc
        
    def set_cumulative_recall(self, cumulative_recall : float) -> None:
        self._cumulative_recall : float = cumulative_recall        
       
def canonical_swap(id1: int, id2: int) -> Tuple[int, int]:
    """Returns the identifiers in canonical order

    Args:
        id1 (int): ID1
        id2 (int): ID2

    Returns:
        Tuple[int, int]: IDs tuple in canonical order (ID1 < ID2)
    """
    if id2 > id1:
        return id1, id2
    else:
        return id2, id1

def sorted_enumerate(seq, reverse=True):
    return [i for (v, i) in sorted(((v, i) for (i, v) in enumerate(seq)), reverse=reverse)]


def is_infinite(value : float):
    return math.isinf(value) and value > 0

def reverse_data_indexing(data : Data) -> Data:
    """Returns a new data model based upon the given data model with reversed indexing of the datasets
    Args:
        data (Data): input dat a model

    Returns:
        Data : New Data Module with reversed indexing
    """
    return Data(dataset_1 = data.dataset_2,
                id_column_name_1 = data.id_column_name_2,
                attributes_1 = data.attributes_2,
                dataset_name_1 = data.dataset_name_2,
                dataset_2 = data.dataset_1,
                attributes_2 = data.attributes_1,
                id_column_name_2 = data.id_column_name_1,
                dataset_name_2 = data.dataset_name_1,
                ground_truth = data.ground_truth)

def get_class_function_arguments(class_reference, function_name : str) -> List[str]:
    """Returns a list of argument names for requested function of the given class
    Args:
        class_reference: Reference to a class
        function_name (str): Name of the requested function

    Returns:
        List[str] : List of requested function's arguments' names
    """
    if not inspect.isclass(class_reference):
        raise ValueError(f"{class_reference.__name__} class reference is not valid.")

    if not hasattr(class_reference, function_name):
        raise ValueError(f"The class {class_reference.__name__} does not have a function named {function_name}.")

    function_obj = getattr(class_reference, function_name)
    if not inspect.isfunction(function_obj):
        raise ValueError(f"The provided name {function_name} does not correspond to a function in class '{class_reference.__name__}'.")

    function_signature = inspect.signature(function_obj)
    argument_names = list(function_signature.parameters.keys())[1:]

    return argument_names

def new_dictionary_from_keys(dictionary : dict, keys : list) -> dict:
    """Returns a subset of the given dictionary including only the given keys.
       Unrecognized keys are not included.
    Args:
        dictionary (dict): Target dictionary
        keys (list): Keys to keep

    Returns:
        dict : Subset of the given dictionary including only the requested keys
    """
    new_dictionary : dict = {key: dictionary[key] for key in keys if key in dictionary}
    return new_dictionary


def has_duplicate_pairs(pairs : List[Tuple[float, int, int]]):
    seen_pairs = set()
    for pair in pairs:
        entity : int = pair[1]
        candidate : int = pair[2]
        if (entity, candidate) in seen_pairs:
            return True
        seen_pairs.add((entity, candidate))
    return False

            
            
        
        
        
    

        
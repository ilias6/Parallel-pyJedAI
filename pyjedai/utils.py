from collections import defaultdict
import numpy as np
from pyjedai.datamodel import Block
from pyjedai.datamodel import Data
from abc import ABC
from typing import List
import random

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

def get_sorted_blocks_shuffled_entities(blocks):
    """Sorts blocks in alphabetical order based on their token, shuffles the entities of each block, concatenates the result in a list

    Args:
        blocks (Dict[Block]): Dictionary of type token -> Block Instance

    Returns:
        List[Int]: List of shuffled entities of alphabetically, token sorted blocks
    """
    sorted_entities = []
    for _, block in sorted(blocks.items()):
        sorted_entities += random.shuffle(list(block.entities_D1 | block.entities_D2 if block.entities_D2 else block.entities_D1))
    return sorted_entities


class SubsetIndexer(ABC):
    """Stores the indices of retained entities of the initial datasets,
       calculates and stores the mapping of element indices from new to old dataset (id in subset -> id in original)
    """

    def __init__(self):
        self.d1_retained_ids: list[int] = None
        self.d2_retained_ids : list[int] = None

    def __init__(self, blocks: dict, data: Data):
        self.d1_retained_ids: list[int] = None
        self.d2_retained_ids : list[int] = None
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

        if(blocks is None):
            self.d1_retained_ids = list(range(self.num_of_entities_1))

            if(not data.is_dirty_er):
                self.d2_retained_ids = list(range(self.num_of_entities_1, self.num_of_entities_1 + self.num_of_entities_2))
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
            self._counters = 0
            
        for index, entity in enumerate(sorted_entities):
            self._entity_positions[entity][self._counters[entity]] = index
            self._counters[entity] += 1
            
    def get_positions(self, entity: int):
        return self._counters[entity]
            
            
        
        
        
    

        
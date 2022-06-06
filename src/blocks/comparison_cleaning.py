'''
Comparison cleaning methods
'''

import numpy as np
import os, sys

import tqdm
from tqdm import tqdm
import math
from math import log10

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.entities import WorkFlow
from src.utils.enums import WEIGHTING_SCHEME
from src.utils.constants import EMPTY
from src.blocks.utils import create_entity_index
from src.utils.constants import  DISCRETIZATION_FACTOR

class AbstractComparisonCleaning:
    '''
    '''
    _progress_bar = None

    def __init__(self) -> None:

        self._is_dirty_er: bool
        self._num_of_entities_1: int
        self._num_of_entities_2: int = None
        self._num_of_blocks: int
        self._dataset_limit: int # for CC-ER
        self._valid_entities: set() = set()
        self._entity_index: dict
        self._weighting_scheme: str
        self._blocks: dict() # initial blocks
        self.blocks = dict() # blocks after CC

    def process(
            self,
            blocks: dict = None,
            num_of_entities_1: int = None,
            num_of_entities_2: int = None,
            workflow: WorkFlow = None
    ) -> dict:

        if num_of_entities_2:
            self._is_dirty_er = False
        else:
            self._is_dirty_er = True

        if workflow and (blocks is None):
            print("WORKFLOW")
            self._is_dirty_er = workflow.is_dirty_er
            self._entity_index = create_entity_index(workflow.blocks, self._is_dirty_er)
            self._num_of_entities_1 = workflow.num_of_entities_1
            self._num_of_entities = workflow.num_of_entities_1
            if not self._is_dirty_er:
                self._num_of_entities_2 = workflow.num_of_entities_2
                self._num_of_entities += workflow.num_of_entities_2
            self._dataset_limit: int = workflow.num_of_entities_1
            self._num_of_blocks = len(workflow.blocks)
            self._blocks: dict = workflow.blocks
        else:
            self._entity_index = create_entity_index(blocks, self._is_dirty_er)
            self._num_of_entities_1 = num_of_entities_1
            self._num_of_entities = self._num_of_entities_1
            if not self._is_dirty_er:
                self._num_of_entities_2 = num_of_entities_2
                self._num_of_entities += self._num_of_entities_2
            self._dataset_limit: int = num_of_entities_1
            self._num_of_blocks = len(blocks)
            self._blocks: dict = blocks

        self._progress_bar = tqdm(total=2*self._num_of_entities, desc=self._method_name)

        return self._apply_main_processing()

class AbstractMetablocking(AbstractComparisonCleaning):
    """
    Goal: Restructure a redundancy-positive block collection into a new
    one that contains substantially lower number of redundant
    and superfluous comparisons, while maintaining the original number of matching ones
    """

    def __init__(self) -> None:
        super().__init__()

        self._flags: np.array
        self._counters: np.array
        self._comparisons_per_entity: np.array
        self._node_centric: bool
        self._threshold: float
        self._distinct_comparisons: int
        self._comparisons_per_entity: np.array
        self._neighbors: list = list()
        self._retained_neighbors: list = list()
        self._retained_neighbors_weights: list = list()
        self._block_assignments: int = 0
        self.weighting_scheme: str

    def _apply_main_processing(self) -> dict:

        self._counters = np.empty([self._num_of_entities], dtype=float)

        for block_key in self._blocks.keys():
            self._block_assignments += self._blocks[block_key].get_total_block_assignments(self._is_dirty_er)

        self._set_threshold()
        
        return self._prune_edges()

    def _get_weight(self, entity_id: int, neighbor_id: int) -> float:
        ws = self._weighting_scheme
        if ws == 'ARCS' or ws == 'CBS':
            return self._counters[neighbor_id]
        elif ws == 'ECBS':
            return float(
                self._counters[neighbor_id] *
                log10(float(self._num_of_blocks / len(self._entity_index[entity_id]))) *
                log10(float(self._num_of_blocks / len(self._entity_index[neighbor_id])))
            )
        elif ws == 'JS':
            return self._counters[neighbor_id] / (len(self._entity_index[entity_id]) + len(self._entity_index[neighbor_id]) - self._counters[neighbor_id])
        elif ws == 'EJS':
            probability = self._counters[neighbor_id] / (len(self._entity_index[entity_id]) + len(self._entity_index[neighbor_id]) - self._counters[neighbor_id])
            return float(probability * log10(self._distinct_comparisons / self._comparisons_per_entity[entity_id]) * log10(self._distinct_comparisons / self._comparisons_per_entity[neighbor_id]))
        elif ws == 'PEARSON_X2':
            # TODO: ChiSquared
            pass
        else:
            # TODO: Error handling
            print('This weighting scheme does not exist')
        
    def _normalize_neighbor_entities(self, block_key: str, entity_id: int) -> None:
        self._neighbors.clear()
        if self._is_dirty_er:
            if not self._node_centric:
                for neighbor_id in self._blocks[block_key].entities_D1:
                    if neighbor_id < entity_id:
                        self._neighbors.append(neighbor_id)
            else:
                for neighbor_id in self._blocks[block_key].entities_D1:
                    if neighbor_id != entity_id:
                        self._neighbors.append(neighbor_id)
        else:
            if entity_id < self._dataset_limit:
                for original_id in self._blocks[block_key].entities_D2:
                    self._neighbors.append(original_id)
            else:
                for original_id in self._blocks[block_key].entities_D1:
                    self._neighbors.append(original_id)

    def _discretize_comparison_weight(self, weight: float) -> int:
        return int(weight * DISCRETIZATION_FACTOR)

    def _set_statistics(self) -> None:
        self._distinct_comparisons = 0
        self._comparisons_per_entity = np.empty([self._num_of_entities], dtype=float)
        distinct_neighbors = set()

        for entity_id in range(0, self._num_of_entities, 1):
            associated_blocks = self._entity_index[entity_id]
            if len(associated_blocks) != 0:
                distinct_neighbors.clear()
                for block_id in associated_blocks:
                    for neighbor_id in self._get_neighbor_entities(block_id, entity_id):
                        distinct_neighbors.add(neighbor_id)
                self._comparisons_per_entity[entity_id] = len(distinct_neighbors)

                if self._is_dirty_er:
                    self._comparisons_per_entity[entity_id] -= 1

                self._distinct_comparisons += self._comparisons_per_entity[entity_id]
        self._distinct_comparisons /= 2

    def _get_neighbor_entities(self, block_id: int, entity_id: int) -> set:
        if not self._is_dirty_er and entity_id < self._dataset_limit:
            return self._blocks[block_id].entities_D2
        return self._blocks[block_id].entities_D1


class WeightedEdgePruning(AbstractMetablocking):

    _method_name = "Weighted Edge Pruning"
    _method_info = ": a Meta-blocking method that retains all comparisons \
                that have a weight higher than the average edge weight in the blocking graph."

    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__()
        self._weighting_scheme = weighting_scheme
        self._node_centric = False
        self._num_of_edges: float

    def _prune_edges(self) -> dict:
        for i in range(0, self._num_of_entities):
            self._process_entity(i)
            self._verify_valid_entities(i)
            self._progress_bar.update(1)

        return self.blocks

    def _process_entity(self, entity_id: int):
        self._valid_entities.clear()
        self._flags = np.empty([self._num_of_entities], dtype=int)
        self._flags[:] = EMPTY
        associated_blocks = self._entity_index[entity_id]

        if len(associated_blocks) == 0:
            print("No associated blocks")
            return

        for block_id in associated_blocks:
            if self.weighting_scheme == 'ARCS':
                block_comparisons = self._blocks[block_id].get_num_of_comparisons(self._is_dirty_er)
            self._normalize_neighbor_entities(block_id, entity_id)
            for neighbor_id in self._neighbors:
                if self._flags[neighbor_id] != entity_id:
                    self._counters[neighbor_id] = 0
                    self._flags[neighbor_id] = entity_id

                if self.weighting_scheme == 'ARCS':
                    self._counters[neighbor_id] += 1/block_comparisons
                else:
                    self._counters[neighbor_id] += 1
                self._valid_entities.add(neighbor_id)

    def _update_threshold(self, entity_id: int) -> None:
        self._num_of_edges += len(self._valid_entities)
        for neighbor_id in self._valid_entities:
            self._threshold += super()._get_weight(entity_id, neighbor_id)

    def _set_threshold(self):
        self._num_of_edges = 0.0
        self._threshold = 0.0

        for i in range(0, self._num_of_entities):
            self._process_entity(i)
            self._update_threshold(i)
            self._progress_bar.update(1)

        self._threshold /= self._num_of_edges

    def _verify_valid_entities(self, entity_id: int) -> None:
        self._retained_neighbors.clear()
        self._retained_neighbors_weights.clear()
        for neighbor_id in self._valid_entities:
            weight = self._get_weight(entity_id, neighbor_id)
            if self._threshold <= weight:
                self._retained_neighbors.append(neighbor_id)
                self._retained_neighbors_weights.append(self._discretize_comparison_weight(weight))
        if len(self._retained_neighbors) > 0:
            self.blocks[entity_id] = self._retained_neighbors.copy()

class WeightedNodePruning(WeightedEdgePruning):
    def __init__(self) -> None:
        super().__init__()
        pass    
    pass

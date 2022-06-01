import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.entities import WorkFlow
from src.utils.enums import WEIGHTING_SCHEME, EMPTY
from src.blocks.utils import create_entity_index
from src.utils.constants import  DISCRETIZATION_FACTOR

class AbstractComparisonCleaning:

    # _entity_index: dict
    # _is_dirty_er: bool
    # _dataset_lim: int
    # _num_of_blocks: int
    # _num_of_entities: int

    def __init__(self) -> None:

        self._is_dirty_er: bool
        self._num_of_entities_1: int
        self._num_of_entities_2: int = None
        self._num_of_blocks: int
        self._dataset_limit: int # for CC-ER
        self._valid_entities: set()
        self._entity_index: dict
        self._weighting_scheme: str

    def process(self, blocks: dict) -> dict:
        self._entity_index, dataset_specs = create_entity_index(blocks, self._is_dirty_er)
        self._num_of_entities_1 = dataset_specs['num_of_entities_1']
        self._num_of_entities = self._num_of_entities_1
        if not self._is_dirty_er:
            self._num_of_entities_2 = dataset_specs['num_of_entities_2']
            self._num_of_entities += self._num_of_entities_2
        self._num_of_blocks = dataset_specs['num_of_blocks']
        self._dataset_limit: int # for CC-ER
        self.blocks: dict = blocks

        return self._apply_main_processing()

    def _apply_main_processing(self) -> dict:
        pass

    def _add_decomposed_block(self, entity_id: int, neighbors: set, neighbors_weights: set, new_blocks: dict) -> None:

        if len(neighbors) == 0:
            return
        
        entity_ids_1 = self._replicate_id(entity_id, len(neighbors))
        new_blocks.setdefault()

    def _replicate_id(self, entity_id, times) -> np.array:
        array = np.empty([times])
        array[:] = entity_id
        return array 

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

        # All dicts will be {int -> np.array}
        self._neighbors: set = set()
        self._retained_neighbors: set = set()
        self._retained_neighbors_weights: set = set()

        self.weighting_scheme: str

    def _apply_main_processing(self, blocks: dict) -> dict:

        block_assignments = 0
        counters = np.empty([self._num_of_entities], dtype=float)

        for block_key in blocks:
            block_assignments += blocks[block_key].get_total_block_assignments(self._is_dirty_er)

        self._set_threshold()

        return self._prune_edges()

    def _get_weight(self, entity_id: int, neighbor_id: int) -> float:
        ws = self._weighting_scheme
        if ws == 'ARCS' or ws == 'CBS':
            return self.counters[neighbor_id]
        elif ws == 'ECBS':
            # TODO
            pass
        elif ws == 'JS':
            # TODO
            pass
        elif ws == 'EJS':
            # TODO
            pass
        elif ws == 'PEARSON_X2':
            # TODO
            pass
        else:
            # TODO error
            print('This weighting scheme does not exist')
        
    def _normalize_neighbour_entities(self, block_key: str, entity_id: int) -> None:
        self._neighbors.clear()
        if self._is_dirty_er:
            if not self._node_centric:
                for neighbor_id in self.blocks[block_key].entities_D1:
                    if neighbor_id < entity_id:
                        self._neighbors.add(neighbor_id)
            else:
                for neighbor_id in self.blocks[block_key].entities_D1:
                    if neighbor_id != entity_id:
                        self._neighbors.add(neighbor_id)
        else:
            if entity_id < self._dataset_limit:
                for original_id in self.blocks[block_key].entities_D2:
                    self._neighbors.add(original_id)
            else:
                for original_id in self.blocks[block_key].entities_D1:
                    self._neighbors.add(original_id)

    def _discretize_comparison_weight(self, weight: float) -> int:
        return (int) weight * DISCRETIZATION_FACTOR

    def _prune_edges(self) -> dict:
        pass

    def _set_threshold(self) -> None:
        pass

class WeightedEdgePruning(AbstractMetablocking):

    _method_name = "Weighted Edge Pruning"
    _method_info = ": a Meta-blocking method that retains all comparisons \
                that have a weight higher than the average edge weight in the blocking graph."

    def __init__(self, weighting_scheme: str, is_dirty_er: bool) -> None:
        super().__init__()
        self._is_dirty_er = is_dirty_er
        self._weighting_scheme = weighting_scheme
        self._node_centric = False
        self._num_of_edges: float = 0.0

    def _prune_edges(self) -> dict:
        new_blocks = dict()
        # TODO ARCS
        for i in range(0, self._num_of_entities):
            self._process_entity(i)
            self._verify_valid_entities(i, new_blocks)

        return new_blocks

    def _process_entity(self, entity_id: int):
        self._valid_entities.clear()
        flags = np.empty([self._num_of_entities], dtype=int)
        flags[:] = EMPTY

        associated_blocks = self._entity_index[entity_id]

        if len(associated_blocks) == 0:
            return

        for block_id in associated_blocks:
            self._normalize_neighbour_entities(block_id, entity_id)
            for neighbor_id in self._neighbors:
                if self._flags[neighbor_id] != entity_id:
                    self._counters[neighbor_id] = 0
                    self._flags[neighbor_id] = entity_id

                self._counters[neighbor_id] += 1
                self._valid_entities.add(neighbor_id)

    def _update_threshold(self, entity_id: int) -> None:
        self._num_of_edges += len(self._valid_entities)

        for id in self._valid_entities:
            self._threshold += super()._get_weight(entity_id, id)

    def _set_threshold(self):
        self._num_of_edges = 0
        self._threshold = 0

        # TODO: ARCS
        # Java inline if ??

        for i in range(0, self._num_of_entities):
            self._process_entity(i)
            self._update_threshold(i)

        self._threshold /= self._num_of_edges

    def _verify_valid_entities(self, entity_id: int, new_blocks: dict) -> None:
        self._retained_neighbors.clear()
        self._retained_neighbors_weights.clear()

        if not self._is_dirty_er:
            for neighbor_id in self._valid_entities:
                weight = self._get_weight(entity_id, neighbor_id)
                if self._threshold <= weight:
                    self._retained_neighbors.add(neighbor_id)
                    self._retained_neighbors_weights.add(self._discretize_comparison_weight(weight))
            
            self._add_decomposed_block(entity_id, self._retained_neighbors, self._retained_neighbors_weights, new_blocks)
        else:
            pass

class WeightedNodePruning(WeightedEdgePruning):
    def __init__(self) -> None:
        super().__init__()
        pass    
    pass

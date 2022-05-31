import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.entities import WorkFlow
from src.utils.enums import WEIGHTING_SCHEME, EMPTY
from src.blocks.utils import create_entity_index

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

        return self._apply_main_processing()

    def _apply_main_processing(self) -> dict:
        pass


class AbstractMetablocking(AbstractComparisonCleaning):
    """
    Goal: Restructure a redundancy-positive block collection into a new
    one that contains substantially lower number of redundant
    and superfluous comparisons, while maintaining the original number of matching ones
    """

    def __init__(self) -> None:
        super().__init__()

        self.flags: np.array
        self.counters: np.array
        self.comparisons_per_entity: np.array

        # All dicts will be {int -> np.array}
        self.neighbors: dict = dict()
        self.retained_neighbors: dict = dict()
        self.retained_neighbors_weights: dict = dict()

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
        self._node_centric: bool = False
        self._num_of_edges: float = 0.0

    def _prune_edges(self):
        new_blocks = {}


        pass
    
    def _process_entity(self, entity_id: int):

        self.valid_entities = set()
        flags = np.empty((self._num_of_entities))
        flags[:] = EMPTY

    def _update_threshold(self, entity_id: int) -> None:
        self._num_of_edges += len(self._valid_entities)

        for id in self._valid_entities:
            self._threshold += super()._get_weight(entity_id, id)

    def _set_threshold(self):
        self._num_of_edges = 0
        self._threshold = 0

        # TODO: ARCS

        for i in range(0, self._num_of_entities):
            self._process_entity(i)
            self._update_threshold(i)

        self._threshold /= self.num_of_edges







class WeightedNodePruning(WeightedEdgePruning):
    def __init__(self) -> None:
        super().__init__()
        pass    
    pass
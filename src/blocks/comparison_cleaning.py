from core.entities import WorkFlow
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.enums import WEIGHTING_SCHEME, EMPTY

class AbstractComparisonCleaning:

    def __init__(self, is_dirty_er, num_of_entities, num_of_blocks, dataset_limit) -> None:
        self._is_dirty_er: bool = is_dirty_er
        self._num_of_entities: int = num_of_entities
        self._num_of_blocks: int = num_of_blocks
        self._dataset_limit: int = dataset_limit # for CC-ER
        self.valid_entities: dict = dict()

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

    def process(self, workflow: WorkFlow) -> dict:

        blocks = workflow.blocks
        block_assignments = 0
        counters = np.array([self._num_of_entities])

        for block_key in blocks:
            block_assignments += blocks[block_key].get_cardinality()

        self.set_threshold()

        return self.prune_edges()

    def prune_edges(self):
        pass
    
    def set_threshold(self):
        pass

class WeightedEdgePruning(AbstractMetablocking, WEIGHTING_SCHEME):

    _method_name = "Weighted Edge Pruning"
    _method_info = ": a Meta-blocking method that retains all comparisons \
                that have a weight higher than the average edge weight in the blocking graph."

    def __init__(self) -> None:
        super().__init__()
        self.num_of_edges: float
        self.node_centric: bool = False

    def prune_edges(self):
        new_blocks = {}


        pass
    
    def process_entity(self, entity_id: int):

        self.valid_entities.clear()
        flags = np.empty((self._num_of_entities))
        flags[:] = EMPTY
        associated_blocks = workflow.


    def set_threshold(self):
        pass

class WeightedNodePruning(WeightedEdgePruning):
    def __init__(self) -> None:
        super().__init__()
        pass    
    pass
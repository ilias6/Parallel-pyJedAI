import numpy as np

class AbstractComparisonCleaning:

    def __init__(self) -> None:
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

        self.neighbors = []
        self.retained_neighbors = []
        self.retained_neighbors_weights = []

        self.weighting_scheme: str



class WeightedEdgePruning(AbstractMetablocking):
    def __init__(self) -> None:
        super().__init__()
        pass
    pass

class WeightedNodePruning(WeightedEdgePruning):
    def __init__(self) -> None:
        super().__init__()
        pass    
    pass
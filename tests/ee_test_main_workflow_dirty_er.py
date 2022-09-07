"""
All tests for the main pyJedAI workflow.
Qaution:
- Only Dirty ER tests.
- Methods called with their default values.
"""

import os
import sys
import pandas as pd
import os
import sys
import pandas as pd
import networkx
from networkx import draw, Graph
import logging as log

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

""" Test Datamodel
"""
from pyjedai.datamodel import Data

dirty_data = Data(
    dataset_1=pd.read_csv("data/cora/cora.csv", sep='|').head(100),
    id_column_name_1='Entity Id',
    ground_truth=pd.read_csv("data/cora/cora_gt.csv", sep='|', header=None),
    attributes_1=['Entity Id', 'author', 'title']
)
dirty_data.process()
dirty_data.print_specs()

def test_datamodel_dirty_er():
    assert dirty_data is not None

from pyjedai.block_building import (
    StandardBlocking,
    QGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking,
    ExtendedQGramsBlocking
)
block_building_methods = [
    StandardBlocking,
    QGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking,
    ExtendedQGramsBlocking
]

from pyjedai.block_cleaning import BlockFiltering, BlockPurging
block_cleaning_methods = [
    BlockFiltering, 
    BlockPurging, 
    None # Optional workflow step
]

from pyjedai.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
    BLAST,
    ReciprocalCardinalityNodePruning,
    ReciprocalWeightedNodePruning,
    ComparisonPropagation
)
comparison_cleaning_methods = [
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
    BLAST,
    ReciprocalCardinalityNodePruning,
    ReciprocalWeightedNodePruning,
    ComparisonPropagation,
    None # Optional workflow step
]

from pyjedai.matching import EntityMatching
entity_matching_methods = [EntityMatching]

from pyjedai.clustering import ConnectedComponentsClustering
clustering_methods = [ConnectedComponentsClustering]

for block_building in block_building_methods:
    for block_cleaning in block_cleaning_methods:
        for comparison_cleaning in comparison_cleaning_methods:
            for matching in entity_matching_methods:
                for clustering in clustering_methods:
                    """ Block Building
                    """
                    blocks = block_building().build_blocks(dirty_data, tqdm_disable=True)
                    def test_block_building():
                        assert blocks is not None
                    
                    """ Block Cleaning - Optional
                    """
                    clean_blocks = None
                    if block_cleaning is not None:
                        clean_blocks = block_cleaning().process(blocks, dirty_data, tqdm_disable=True)
                        def test_block_cleaning_method():
                            assert clean_blocks is not None

                    """ Comparison Cleaning - Optional
                    """
                    candidate_pairs_blocks = None
                    if comparison_cleaning is not None:
                        candidate_pairs_blocks = comparison_cleaning().process(
                            clean_blocks if clean_blocks is not None else blocks, dirty_data, tqdm_disable=True)
                        def test_comparison_cleaning_method():
                            assert candidate_pairs_blocks is not None
                    
                    pairs_graph = matching(metric="sorensen_dice").predict(blocks, dirty_data, tqdm_disable=True)
                    def test_matching_method(graph):
                        assert graph is not None
                    if candidate_pairs_blocks: 
                        test_matching_method(matching(metric="sorensen_dice").predict(candidate_pairs_blocks, dirty_data, tqdm_disable=True))
                    if clean_blocks: 
                        test_matching_method(matching(metric="sorensen_dice").predict(candidate_pairs_blocks, dirty_data, tqdm_disable=True))

                    clusters = clustering().process(pairs_graph)
                    def test_clustering_method():
                        assert clusters is not None
                        
                    

import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from pyjedai.block_building import (ExtendedQGramsBlocking,
#                                     ExtendedSuffixArraysBlocking,
#                                     QGramsBlocking, StandardBlocking,
#                                     SuffixArraysBlocking)
# from pyjedai.block_cleaning import BlockFiltering, BlockPurging
# from pyjedai.clustering import ConnectedComponentsClustering
# from pyjedai.comparison_cleaning import (BLAST, CardinalityEdgePruning,
#                                          CardinalityNodePruning,
#                                          ComparisonPropagation,
#                                          ReciprocalCardinalityNodePruning,
#                                          ReciprocalWeightedNodePruning,
#                                          WeightedEdgePruning,
#                                          WeightedNodePruning)
# from pyjedai.datamodel import Data
# from pyjedai.matching import EntityMatching
# from pyjedai.workflow import WorkFlow, compare_workflows

directory = r'../data/ccer'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if "D" in f:
        print(f)
        os.path.join(directory, f)
        data = Data(
            dataset_1=pd.read_csv("data/D2/abt.csv", sep='|', engine='python').astype(str),
            attributes_1=['id', 'name', 'description'],
            id_column_name_1='id',
            dataset_2=pd.read_csv("data/D2/buy.csv", sep='|', engine='python').astype(str),
            attributes_2=['id', 'name', 'description'],
            id_column_name_2='id',
            ground_truth=pd.read_csv("data/D2/gt.csv", sep='|', engine='python')
        )

'''
 Main workflow for Clean-Clean ER
'''
# --- Libs import --- #
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.tokenizer import Tokenizer, cora_text_cleaning_method
from src.blocks.building import StandardBlocking, QGramsBlocking
from src.blocks.cleaning import BlockFiltering
from src.blocks.comparison_cleaning import WeightedEdgePruning
from src.core.entities import WorkFlow
from src.blocks.utils import print_blocks, print_candidate_pairs


# --- 1. Read the dataset --- #

dataset_1 = pd.read_csv(
    "../data/cora/cora.csv",
    usecols=['title'],
    # nrows=10,
    sep='|'
)

dataset_2 = pd.read_csv(
    "../data/cora/cora.csv",
    usecols=['author'],
    nrows=100,
    sep='|'
)

ground_truth = pd.read_csv("../data/cora/cora_gt.csv", sep='|')

is_dirty_er = False

w = WorkFlow(
    dataset_1=dataset_1,
    dataset_2=dataset_2,
    ground_truth=ground_truth
)

# --- 2. Block Building techniques --- #

SB = StandardBlocking(text_cleaning_method=cora_text_cleaning_method)
blocks = SB.build_blocks(workflow=w)

# blocks = QGramsBlocking(
#     qgrams=2,
#     text_cleaning_method=cora_text_cleaning_method
# ).build_blocks(dataset_1, dataset_2)


# --- 4. Block Filtering --- #
BF = BlockFiltering(is_dirty_er, ratio=0.9)
blocks = BF.process(blocks=SB.blocks, dataset_lim=SB._dataset_lim)

print_blocks(blocks, is_dirty_er)

# --- META-Blocking -- #

WE = WeightedEdgePruning()
candidate_pairs_blocks = WE.process(blocks, SB._num_of_entities_1, SB._num_of_entities_2)

print_candidate_pairs(candidate_pairs_blocks)
# --- 5. Comparison Propagation --- #
# --- 6. Jaccard Similarity --- #
# --- 7. Connected Components --- #

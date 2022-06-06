'''
 Main workflow for Dirty ER
'''
# --- Libs import --- #
from html import entities
import os
import sys
import pandas as pd
from pyrsistent import b

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.tokenizer import Tokenizer, cora_text_cleaning_method
from src.blocks.building import StandardBlocking, QGramsBlocking
from src.blocks.cleaning import BlockFiltering
from src.blocks.comparison_cleaning import WeightedEdgePruning
from src.core.entities import WorkFlow
from src.blocks.utils import print_blocks, print_candidate_pairs

# --- 1. Read the dataset --- #

dataset = pd.read_csv(
    "../data/cora/cora.csv",
    usecols=['title', 'author'],
    # nrows=10,
    sep='|'
)

ground_truth = pd.read_csv("../data/cora/cora_gt.csv", sep='|')

# Create WorkFlow

w = WorkFlow(
    dataset_1=dataset,
    ground_truth=ground_truth
)

is_dirty_er = True
# --- 2. Block Building techniques --- #

SB = StandardBlocking(text_cleaning_method=cora_text_cleaning_method)
SB.build_blocks(workflow=w)

# print(blocks)

# blocks = QGramsBlocking(
#     qgrams=2,
#     text_cleaning_method=cora_text_cleaning_method
# ).build_blocks(workflow=w)

# print(blocks)

# --- 4. Block Filtering --- #

BF = BlockFiltering(is_dirty_er, ratio=0.9)
blocks = BF.process(blocks=SB.blocks, dataset_lim=SB._dataset_lim)


print_blocks(blocks, is_dirty_er)


# --- META-Blocking -- #

WE = WeightedEdgePruning()
candidate_pairs_blocks = WE.process(blocks, SB._num_of_entities_1)

print_candidate_pairs(candidate_pairs_blocks)

# --- 5. Comparison Propagation --- #
# --- 6. Jaccard Similarity --- #
# --- 7. Connected Components --- #

'''
 Main workflow for Dirty ER
'''
# --- Libs import --- #
from html import entities
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.tokenizer import Tokenizer, cora_text_cleaning_method
from src.blocks.building import StandardBlocking, QGramsBlocking
from src.blocks.cleaning import BlockFiltering
from src.core.entities import WorkFlow

# --- 1. Read the dataset --- #

dataset = pd.read_csv(
    "../data/cora/cora.csv",
    usecols=['author'],
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

blocks = StandardBlocking(
    text_cleaning_method=cora_text_cleaning_method
).build_blocks(workflow=w)

# print(blocks)

blocks = QGramsBlocking(
    qgrams=2,
    text_cleaning_method=cora_text_cleaning_method
).build_blocks(workflow=w)

# print(blocks)

# --- 4. Block Filtering --- #

blocks = BlockFiltering(ratio=0.6).process(workflow=w)


# --- META-Blocking -- #

# blocks = WeightedEdgePruning().process(workflow)


for k, b in blocks.items():
    b.verbose(is_dirty_er)

# --- 5. Comparison Propagation --- #
# --- 6. Jaccard Similarity --- #
# --- 7. Connected Components --- #

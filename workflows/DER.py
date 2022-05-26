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

# print(tokens)

# Create WorkFlow

workflow = WorkFlow(
    is_dirty_er=True,
    dataset_1=dataset,
    ground_truth=ground_truth
)

# --- 2. Block Building techniques --- #

StandardBlocking(text_cleaning_method=cora_text_cleaning_method).build_blocks(workflow)

# print(blocks)

QGramsBlocking(
    qgrams=2,
    text_cleaning_method=cora_text_cleaning_method
).build_blocks(workflow)

# print(blocks)

# --- 4. Block Filtering --- #

BlockFiltering(ratio=0.6).process(workflow)


# --- META-Blocking -- #

# WeightedEdgePruning(workflow).process(workflow)


# for k,b in blocks.items():
#     b.verbose()

# --- 5. Comparison Propagation --- #
# --- 6. Jaccard Similarity --- #
# --- 7. Connected Components --- #

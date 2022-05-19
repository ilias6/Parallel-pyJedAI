'''
 Main workflow for Dirty ER
'''
# --- Libs import --- #
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.tokenizer import Tokenizer, cora_text_cleaning_method
from src.blocks.building import StandardBlocking, QGramsBlocking
from src.blocks.cleaning import BlockFiltering

# --- 1. Read the dataset --- #

dataset = pd.read_csv(
    "../data/cora/cora.csv",
    usecols=['title', 'author'],
    sep='|'
)

ground_truth = pd.read_csv("../data/cora/cora_gt.csv", sep='|')

# print(tokens)

# --- 2. Block Building techniques --- #

standard_blocking = StandardBlocking(text_cleaning_method=cora_text_cleaning_method)
blocks = standard_blocking.build_blocks(dataset)

print(blocks)

# qgrams_blocking = QGramsBlocking(
#     qgrams=2,
#     text_cleaning_method=cora_text_cleaning_method
# )
# blocks = qgrams_blocking.build_blocks(dataset)

# print(blocks)

# --- 4. Block Filtering --- #
# block_filtering = BlockFiltering()
# blocks = block_filtering.process(blocks)

# --- 5. Comparison Propagation --- #
# --- 6. Jaccard Similarity --- #
# --- 7. Connected Components --- #

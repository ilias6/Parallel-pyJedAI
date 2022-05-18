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

# --- 1. Read the dataset --- #

dataset = pd.read_csv(
    "../data/cora/cora.csv",
    usecols=['title'],
    sep='|'
)

ground_truth = pd.read_csv("../data/cora/cora_gt.csv", sep='|')

# print(tokens)

# --- 2. Block Building techniques --- #

standard_blocking = StandardBlocking()
blocks = standard_blocking.build_blocks(dataset)


qgrams_blocking = QGramsBlocking(
    qgrams=2,
    is_char_tokenization=True,
    text_cleaning_method=cora_text_cleaning_method
)
blocks = qgrams_blocking.build_blocks(dataset)

# print(blocks)

# --- 4. Block Filtering --- #


# --- 5. Comparison Propagation --- #
# --- 6. Jaccard Similarity --- #
# --- 7. Connected Components --- #

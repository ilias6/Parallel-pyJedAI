'''
 Main workflow for Dirty ER
'''
# --- Libs import --- #
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.tokenizer import Tokenizer
from src.blocks.building import StandardBlocking

# --- 1. Read the dataset --- #

dataset = pd.read_csv("../data/cora/cora.csv", sep = '|')
groundtruth = pd.read_csv("../data/cora/cora_gt.csv", sep = '|')

# --- 2. Tokenize techniques --- #
'''
 - Tokens/n-grams
'''

tok = Tokenizer(ngrams=2, is_char_tokenization=False, return_type='list')
tokens = tok.process(dataset['title'])

print(tokens)

# --- 3. Block Building techniques --- #

standard_blocking = StandardBlocking()
# standard_blocking.build_blocks(tokens)

# --- 4. Block Filtering --- #


# --- 5. Comparison Propagation --- #
# --- 6. Jaccard Similarity --- #
# --- 7. Connected Components --- #

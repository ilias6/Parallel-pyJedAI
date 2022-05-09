'''
 Main workflow for Dirty ER
'''

# --- Libs import --- #
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utilities.tokenizer import Tokenizer


# --- 1. Read the dataset -- #
'''
 Cora example
'''

dataset = pd.read_csv("../data/cora/cora.csv", sep = '|')
groundtruth = pd.read_csv("../data/cora/cora_gt.csv", sep = '|')

# --- 2. Tokenize techniques -- #
'''
 - Tokens/n-grams
 - Tfidf/BoW
'''
tok = Tokenizer(qgrams=2, is_char_tokenization=False)
tokens_array = tok.process(dataset['title'])

print(tokens_array)


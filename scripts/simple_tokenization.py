import os
import sys
import pandas as pd
import networkx
from networkx import draw, Graph

from pyjedai.utils import (
    text_cleaning_method,
    print_clusters,
    print_blocks,
    print_candidate_pairs
)
from pyjedai.evaluation import Evaluation, write
from pyjedai.datamodel import Data
d1 = pd.read_csv("../data/ccer/D2/abt.csv", sep='|', engine='python')
d2 = pd.read_csv("../data/ccer/D2/buy.csv", sep='|', engine='python')
gt = pd.read_csv("../data/ccer/D2/gt.csv", sep='|', engine='python')


data = Data(
    dataset_1=d1,
    id_column_name_1='id',
    dataset_2=d2,
    id_column_name_2='id',
    ground_truth=gt,
)

from pyjedai.block_building import StandardBlocking
from pyjedai.block_cleaning import BlockFiltering
from pyjedai.block_cleaning import BlockPurging
from pyjedai.comparison_cleaning import WeightedEdgePruning, WeightedNodePruning, CardinalityEdgePruning, CardinalityNodePruning
from pyjedai.matching import EntityMatching

sb = StandardBlocking()
blocks = sb.build_blocks(data)
sb.evaluate(blocks, with_classification_report=True)

cbbp = BlockPurging()
blocks = cbbp.process(blocks, data, tqdm_disable=False)
cbbp.evaluate(blocks, with_classification_report=True)

bf = BlockFiltering(ratio=0.8)
blocks = bf.process(blocks, data, tqdm_disable=False)

wep = CardinalityEdgePruning(weighting_scheme='X2')
candidate_pairs_blocks = wep.process(blocks, data)
wep.evaluate(candidate_pairs_blocks, with_classification_report=True)


string_metrics = [
    'jaro', 'edit_distance'
]

set_metrics = [
    'cosine', 'dice', 'generalized_jaccard', 'jaccard', 'overlap_coefficient'
]

char_qgram_tokenizers = { 'char_'+ str(i) + 'gram':i for i in range(1, 6) }
word_qgram_tokenizers = { 'word_'+ str(i) + 'gram':i for i in range(1, 6) }
magellan_tokenizers = ['white_space_tokenizer']
tok = [key for key in char_qgram_tokenizers] + [key for key in word_qgram_tokenizers]+magellan_tokenizers

for m in string_metrics+set_metrics:
    for t in tok:
        print("\nM =",m,"\nT =",t)
        EM = EntityMatching(metric=m, 
                            tokenizer = t, 
                            similarity_threshold=0.0)
        pairs_graph = EM.predict(candidate_pairs_blocks, data)
        EM.evaluate(pairs_graph)
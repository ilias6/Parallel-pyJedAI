# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:31:58 2023

@author: G_A.Papadakis
"""

import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.block_building import StandardBlocking
from pyjedai.block_cleaning import BlockFiltering
from pyjedai.comparison_cleaning import WeightedEdgePruning

data_dir = '../data/ccer/D8/'

d1 = pd.read_csv(data_dir+"walmart.csv", sep='|', engine='python', na_filter=False).astype(str)
d2 = pd.read_csv(data_dir+"amazon.csv", sep='|', engine='python', na_filter=False).astype(str)
gt = pd.read_csv(data_dir+"gt.csv", sep='|', engine='python')

if 'aggregate value' in d1.columns.tolist():
    d1.drop(columns=['aggregate value'], inplace=True)

if 'aggregate value' in d2.columns.tolist():
    d2.drop(columns=['aggregate value'], inplace=True)


data = Data(dataset_1=d1,
            id_column_name_1='id',
            dataset_2=d2,
            id_column_name_2='id',
            ground_truth=gt)


sb = StandardBlocking()
blocks = sb.build_blocks(data, tqdm_disable=False)

block_perf = sb.evaluate(blocks)
print(block_perf)
sb.stats(blocks)

bf = BlockFiltering(ratio=0.075)
blocks = bf.process(blocks, data, tqdm_disable=False)

block_perf = bf.evaluate(blocks)
print(block_perf)
bf.stats(blocks)

wep = WeightedEdgePruning(weighting_scheme='ARCS')
candidate_pairs_blocks = wep.process(blocks, data, tqdm_disable=False)
               
block_perf = wep.evaluate(candidate_pairs_blocks)
print(block_perf)


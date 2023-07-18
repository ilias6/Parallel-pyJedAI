# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:31:58 2023

@author: G_A.Papadakis
"""

import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.block_building import StandardBlocking
from pyjedai.block_cleaning import BlockFiltering
from pyjedai.comparison_cleaning import WeightedNodePruning

data_dir = '../data/ccer/D3/'

data = Data(dataset_1=pd.read_csv(data_dir+"amazon.csv", sep='#', engine='python', na_filter=False).astype(str),
                id_column_name_1='id',
                dataset_2=pd.read_csv(data_dir+"gp.csv", sep='#', engine='python', na_filter=False).astype(str),
                id_column_name_2='id',
                ground_truth=pd.read_csv(data_dir+"gt.csv", sep='#', engine='python'))

if 'aggregated value' in data.attributes_1:
    data.dataset_1 = data.dataset_1.drop(columns=['aggregated value'], inplace=True)

if 'aggregated value' in data.attributes_2:
    data.dataset_2 = data.dataset_2.drop(columns=['aggregated value'], inplace=True)

sb = StandardBlocking()
blocks = sb.build_blocks(data, tqdm_disable=False)

block_perf = sb.evaluate(blocks)
print(block_perf)
# sb.stats(blocks)

bf = BlockFiltering(ratio=0.6)
blocks = bf.process(blocks, data, tqdm_disable=False)

block_perf = bf.evaluate(blocks)
print(block_perf)
bf.stats(blocks)

wnp = WeightedNodePruning(weighting_scheme='CBS')
candidate_pairs_blocks = wnp.process(blocks, data, tqdm_disable=False)
            
block_perf = wnp.evaluate(candidate_pairs_blocks)
print(block_perf)
	
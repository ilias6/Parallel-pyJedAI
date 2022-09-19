import os
import sys
import pandas as pd
import networkx
from networkx import (
    draw,
    DiGraph,
    Graph,
)

from .evaluation import Evaluation
from .datamodel import Data
from .vector_based_blocking import EmbeddingsNNBlockBuilding

d1 = pd.read_csv("../../data/D2/abt.csv", sep='|', engine='python').astype(str)
d2 = pd.read_csv("../../data/D2/buy.csv", sep='|', engine='python').astype(str)
gt = pd.read_csv("../../data/D2/gt.csv", sep='|', engine='python')

data = Data(
    dataset_1=d1,
    attributes_1=['id','name','description'],
    id_column_name_1='id',
    dataset_2=d2,
    attributes_2=['id','name','description'],
    id_column_name_2='id',
    ground_truth=gt
)

print("\n\nClean-Clean ER in ABT-BUY dataset:\n")
data.process()
data.print_specs()
emb = EmbeddingsNNBlockBuilding(
    vectorizer='bert',
    similarity_search='faiss'
)
blocks = emb.build_blocks(data)
Evaluation(data).report(blocks, emb.method_configuration())

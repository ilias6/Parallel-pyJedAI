import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from pyjedai.datamodel import Data
from pyjedai.block_building import (
    StandardBlocking,
    QGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking,
    ExtendedQGramsBlocking
)

# ---------------------------------------- #
# --------------- Dirty ER --------------- #
# ---------------------------------------- #

d1 = pd.read_csv("data/cora/cora.csv", sep='|')
gt = pd.read_csv("data/cora/cora_gt.csv", sep='|', header=None)
attr = ['Entity Id', 'author', 'title']

data = Data(
    dataset_1=d1,
    id_column_name_1='Entity Id',
    ground_truth=gt,
    attributes_1=attr
)
data.process()
data.print_specs()

def test_standard_blocking_der():
    assert StandardBlocking().build_blocks(data, tqdm_disable=True) is not None

def test_qgrams_blocking_der():
    assert QGramsBlocking().build_blocks(data, tqdm_disable=True) is not None

def test_extended_qgrams_blocking_der():
    assert ExtendedQGramsBlocking().build_blocks(data, tqdm_disable=True) is not None

# ---------------------------------------------- #
# --------------- Clean-Clean ER --------------- #
# ---------------------------------------------- #

d1 = pd.read_csv("data/D2/abt.csv", sep='|', engine='python').astype(str)
d2 = pd.read_csv("data/D2/buy.csv", sep='|', engine='python').astype(str)
gt = pd.read_csv("data/D2/gt.csv", sep='|', engine='python')

data = Data(
    dataset_1=d1,
    attributes_1=['id', 'name', 'description'],
    id_column_name_1='id',
    dataset_2=d2,
    attributes_2=['id', 'name', 'description'],
    id_column_name_2='id',
    ground_truth=gt
)

print("\n\nClean-Clean ER in ABT-BUY dataset:\n")
data.process()
data.print_specs()

def test_standard_blocking_der():
    assert StandardBlocking().build_blocks(data, tqdm_disable=True) is not None

def test_qgrams_blocking_der():
    assert QGramsBlocking().build_blocks(data, tqdm_disable=True) is not None

def test_extended_qgrams_blocking_der():
    assert ExtendedQGramsBlocking().build_blocks(data, tqdm_disable=True) is not None
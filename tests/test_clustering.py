import os
import sys

import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from pyjedai.block_building import (ExtendedQGramsBlocking,
                                    ExtendedSuffixArraysBlocking,
                                    QGramsBlocking, StandardBlocking,
                                    SuffixArraysBlocking)
from pyjedai.datamodel import Data

dirty_data = Data(
    dataset_1=pd.read_csv("data/der/cora/cora.csv", sep='|'),
    id_column_name_1='Entity Id',
    ground_truth=pd.read_csv("data/der/cora/cora_gt.csv", sep='|', header=None),
)
dirty_data.print_specs()

def test_datamodel_dirty_er():
    assert dirty_data is not None

clean_clean_data = Data(
    dataset_1=pd.read_csv("data/ccer/D2/abt.csv", sep='|', engine='python').astype(str),
    id_column_name_1='id',
    dataset_2=pd.read_csv("data/ccer/D2/buy.csv", sep='|', engine='python').astype(str),
    id_column_name_2='id',
    ground_truth=pd.read_csv("data/ccer/D2/gt.csv", sep='|', engine='python')
)
clean_clean_data.print_specs()
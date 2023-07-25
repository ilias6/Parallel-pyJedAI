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
d1 = pd.read_csv("../data/ccer/schema_matching/authors1.csv")
d2 = pd.read_csv("../data/ccer/schema_matching/authors2.csv")
gt = pd.read_csv("../data/ccer/schema_matching/pairs.csv")


data = Data(
    dataset_1=d1,
    attributes_1=['EID','Authors','Cited by','Title','Year','Source tittle','DOI'],
    id_column_name_1='EID',
    dataset_2=d2,
    attributes_2=['EID','Authors','Cited by','Country','Document Type','City','Access Type','aggregationType'],
    id_column_name_2='EID',
    ground_truth=gt,
)

from pyjedai.schema_matching import ValentineMethodBuilder, ValentineSchemaMatching

sm = ValentineSchemaMatching(ValentineMethodBuilder.coma_matcher(strategy="COMA_OPT"))
sm.process(data)

sm.print_matches()

sm.evaluate()
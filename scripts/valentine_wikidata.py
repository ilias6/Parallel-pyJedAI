import os
import sys
import pandas as pd
import networkx
from networkx import draw, Graph

from valentine.data_sources import DataframeTable
from valentine import valentine_match, valentine_metrics, NotAValentineMatcher
from valentine.algorithms import *

print(" joinable")

df1 = pd.read_json("../data/ccer/schema_matching/Musicians_joinable/musicians_joinable_source.json")
df2 = pd.read_json("../data/ccer/schema_matching/Musicians_joinable/musicians_joinable_target.json")

print(', '.join(df1.columns.tolist()))
print(', '.join(df2.columns.tolist()))

# Instantiate matcher and run
matcher =  Cupid(w_struct = 0.2, leaf_w_struct = 0.2, th_accept = 0.7)
matches = valentine_match(df1, df2, matcher)

for match, sim in matches.items():
    print(match, " - ", sim)
    
print("\n\n semjoinable")

df1 = pd.read_json("../data/ccer/schema_matching/Musicians_semjoinable/musicians_semjoinable_source.json")
df2 = pd.read_json("../data/ccer/schema_matching/Musicians_semjoinable/musicians_semjoinable_target.json")


print(', '.join(df1.columns.tolist()))
print(', '.join(df2.columns.tolist()))

# Instantiate matcher and run
matcher =  Cupid(w_struct = 0.2, leaf_w_struct = 0.2, th_accept = 0.7)
matches = valentine_match(df1, df2, matcher)

for match, sim in matches.items():
    print(match, " - ", sim)
    
print("\n\n unionable")

df1 = pd.read_json("../data/ccer/schema_matching/Musicians_unionable/musicians_unionable_source.json")
df2 = pd.read_json("../data/ccer/schema_matching/Musicians_unionable/musicians_unionable_target.json")

print(', '.join(df1.columns.tolist()))
print(', '.join(df2.columns.tolist()))

# Instantiate matcher and run
matcher =  Cupid(w_struct = 0.2, leaf_w_struct = 0.2, th_accept = 0.7)
matches = valentine_match(df1, df2, matcher)

for match, sim in matches.items():
    print(match, " - ", sim)
    
print("\n\n viewunion")

df1 = pd.read_json("../data/ccer/schema_matching/Musicians_viewunion/musicians_viewunion_source.json")
df2 = pd.read_json("../data/ccer/schema_matching/Musicians_viewunion/musicians_viewunion_target.json")


print(', '.join(df1.columns.tolist()))
print(', '.join(df2.columns.tolist()))

# Instantiate matcher and run
matcher =  Cupid(w_struct = 0.2, leaf_w_struct = 0.2, th_accept = 0.7)
matches = valentine_match(df1, df2, matcher)

for match, sim in matches.items():
    print(match, " - ", sim)


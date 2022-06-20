import tqdm
from tqdm import tqdm
import pandas as pd
import networkx as nx
import os
import sys

from src.core.entities import Data

class ConnectedComponentsClustering:

    def __init__(self, similarity_threshold: float) -> None:
        self.similarity_threshold = similarity_threshold

    def process(self, graph: nx.Graph) -> pd.DataFrame:

        connected_components = nx.connected_components(graph)
        pairs_df = pd.DataFrame(columns=["id1", "id2"])
        num_of_pairs = 1
        for cc in connected_components:
            print("Component: ", cc)
            sorted_component = sorted(cc)

            for id1_index in range(0, len(sorted_component), 1):
                for id2_index in range(id1_index+1, len(sorted_component), 1):
                    pairs_df.loc[num_of_pairs] = [sorted_component[id1_index], sorted_component[id2_index]]
                    num_of_pairs += 1

        return pairs_df
import tqdm
from tqdm import tqdm
import networkx as nx
import os
import sys

from src.core.entities import Data




class ConnectedComponentsClustering:

    def __init__(self, similarity_threshold: float) -> None:
        self.similarity_threshold = similarity_threshold

    def process(self, graph: nx.Graph, data: Data) -> :

        connected_components = nx.connected_components(graph)

        
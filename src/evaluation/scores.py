'''
TODO info
'''

import logging
import os
import sys

import pandas as pd
import nltk
import numpy as np
import tqdm
from tqdm import tqdm

from typing import Dict, List, Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.entities import Block, Data
from src.blocks.utils import drop_single_entity_blocks

class Evaluation:

    def __init__(self, data: Data) -> None:
        self.f1: float
        self.recall: float
        self.precision: float
        self.accuracy: float
        self.num_of_comparisons: int
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_matching_pairs = 0
        self._entity_cluster_index = dict()
        self.data: Data = data

    def report(self, predicted_clusters: list, data: Data) -> None:

        gt = data.ground_truth

        self._inverse_clusters(predicted_clusters)

        for pair in gt.itterrows():
            id1 = pair['id1']
            id2 = pair['id2']

            if self._entity_cluster_index[id1] == self._entity_cluster_index[id2]:
                self.true_positives += 1
            else:
                self.false_negatives += 1

        self.false_positives = self.total_matching_pairs - self.true_positives
        self.true_negatives = self.total_matching_pairs - self.false_positives
        

        # self.accuracy = 
        # self.precision = 
        # self.recall = 
        # self.f1 = 
        self.print_results()

    def print_results(self) -> None:
        print()

    def _inverse_clusters(self, clusters: list) -> dict:

        for cluster, cluster_id in (clusters, range(0, len(clusters))):
            cluster_entities_d1 = 0
            cluster_entities_d2 = 0
            for id in cluster:
                self._entity_cluster_index[id] = cluster_id

                if not self.data.is_dirty_er:
                    if id < self.data.dataset_limit:
                        cluster_entities_d1 += 1
                    else:
                        cluster_entities_d2 += 1

            if self.data.is_dirty_er:
                self.total_matching_pairs += len(cluster)*(len(cluster)-1)/2
            else:
                self.total_matching_pairs += cluster_entities_d1*cluster_entities_d2
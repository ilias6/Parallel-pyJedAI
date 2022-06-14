'''
TODO
'''

import strsimpy
from strsimpy.levenshtein import Levenshtein
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from strsimpy.weighted_levenshtein import WeightedLevenshtein
from strsimpy.damerau import Damerau
from strsimpy.optimal_string_alignment import OptimalStringAlignment
from strsimpy.jaro_winkler import JaroWinkler
from strsimpy.longest_common_subsequence import LongestCommonSubsequence
from strsimpy.metric_lcs import MetricLCS
from strsimpy.ngram import NGram
from strsimpy.qgram import QGram
from strsimpy.overlap_coefficient import OverlapCoefficient
from strsimpy.cosine import Cosine
from strsimpy.jaccard import Jaccard
from strsimpy.sorensen_dice import SorensenDice
from strsimpy import SIFT4

import tqdm
from tqdm import tqdm
import networkx
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.entities import Block, Data
from src.blocks.utils import drop_single_entity_blocks, create_entity_index, print_blocks


class EntityMatching:
    '''
    TODO
    '''
    _method_name: str = "Entity Matching"
    _method_info: str = ": Calculates similarity from 0. to 1. for all blocks"

    def __init__(self, metric: str, ngram: int = 2) -> None:
        self.data: Data
        self.pairs: networkx.Graph
        self.metric = metric
        self.ngram: int = 2


    def predict(self, blocks: dict, data: Data) -> list:
        '''
        TODO
        '''
        if len(blocks) == 0:
            # TODO: Error
            return None
        self.data = data
        self.pairs = networkx.Graph()
        all_blocks = list(blocks.values())
        self._progress_bar = tqdm(total=len(all_blocks), desc=self._method_name+" ("+self.metric+")")

        if isinstance(all_blocks[0], Block):
            self._predict_raw_blocks(blocks)
        elif isinstance(all_blocks[0], set):
            self._predict_prunned_blocks(blocks)
        else:
            # TODO: Error
            pass
        return self.pairs

    def _predict_raw_blocks(self, blocks: dict) -> None:
        for _, block in blocks.items():
            if self.data.is_dirty_er:
                entities_array = list(block.entities_D1)
                for entity_id_1 in range(0, len(entities_array), 1):
                    for entity_id_2 in range(entity_id_1+1, len(entities_array), 1):
                        self.pairs.add_edge(
                            entity_id_1, entity_id_2, 
                            weight=self._similarity(self.data.entities_d1[entity_id_1], self.data.entities_d1[entity_id_2])
                        )
            else:
                for entity_id_1 in block.entities_D1:
                    for entity_id_2 in block.entities_D2:
                        self.pairs.add_edge(
                            entity_id_1, entity_id_2,
                            weight=self._similarity(
                                self.data.entities_d1[entity_id_1], self.data.entities_d2[entity_id_2]
                            )
                        )
            self._progress_bar.update(1)

    def _predict_prunned_blocks(self, blocks: dict) -> None:
        for entity_id, candidates in blocks.items():
            for candidate_id in candidates:
                self.pairs.add_edge(
                    entity_id, candidate_id,
                    weight=self._similarity(
                        self.data.entities_d1[entity_id], 
                        self.data.entities_d2[candidate_id]
                    )
                )
            self._progress_bar.update(1)

    def _similarity(self, entity_1: str, entity_2: str) -> float:

        if self.metric == 'levenshtein' or self.metric == 'edit_distance':
            return Levenshtein().distance(entity_1, entity_2)
        elif self.metric == 'nlevenshtein':
            return NormalizedLevenshtein().distance(entity_1, entity_2)
        elif self.metric == 'jaro_winkler':
            return JaroWinkler().distance(entity_1, entity_2)
        elif self.metric == 'metric_lcs':
            return MetricLCS().distance(entity_1, entity_2)
        elif self.metric == 'ngram':
            return NGram(self.ngram).distance(entity_1, entity_2)
        elif self.metric == 'cosine':
            cosine = Cosine(self.ngram)
            return cosine.similarity_profiles(cosine.get_profile(entity_1), cosine.get_profile(entity_2))
        elif self.metric == 'jaccard':
            return Jaccard(self.ngram).distance(entity_1, entity_2)
        elif self.metric == 'sorensen_dice':
            return SorensenDice().distance(entity_1, entity_2)
        elif self.metric == 'overlap_coefficient':
            return OverlapCoefficient().distance(entity_1, entity_2)


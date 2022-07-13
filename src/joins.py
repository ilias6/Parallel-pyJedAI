import pandas as pd
import tqdm
from tqdm import tqdm
import networkx
import os
import sys

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

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datamodel import Data
from utils.constants import EMPTY


class AbstractJoin:
    
    def __init__(self, metric: str, qgrams: int = None) -> None:
        self.metric = metric
        self.qgrams = qgrams
        if self.metric == 'levenshtein' or self.metric == 'edit_distance':
            self._metric = Levenshtein().distance
        elif self.metric == 'nlevenshtein':
            self._metric = NormalizedLevenshtein().distance
        elif self.metric == 'jaro_winkler':
            self._metric = JaroWinkler().distance
        elif self.metric == 'metric_lcs':
            self._metric = MetricLCS().distance
        elif self.metric == 'qgram':
            self._metric = NGram(self.qgrams).distance
        # elif self.metric == 'cosine':
        #     cosine = Cosine(self.qgram)
        #     self._metric = cosine.similarity_profiles(cosine.get_profile(entity_1), cosine.get_profile(entity_2))
        elif self.metric == 'jaccard':
            self._metric = Jaccard(self.qgrams).distance
        elif self.metric == 'sorensen_dice':
            self._metric = SorensenDice().distance
        elif self.metric == 'overlap_coefficient':
            self._metric = OverlapCoefficient().distance
    
    def _tokenize_entity(self, entity: str) -> list:
        
        if self.tokenization == 'qgrams':
            return [' '.join(grams) for grams in nltk.ngrams(entity, n=self.qgrams)]
        elif self.tokenization == 'standard':
            return entity.split()
        elif self.tokenization == 'suffix_arrays':
            return [' '.join(grams) for grams in nltk.ngrams(entity, n=self.qgrams)]
        else:
            print("Tokenization not found")
            # TODO error             
    
    def _create_entity_index(self, entities: pd.DataFrame, attributes: any = None) -> dict:
        if attributes and isinstance(attributes, dict):
            attributes = list(attributes.keys())
        
        index = {}
        for i in range(0, self.data.num_of_entities_1, 1):
            record = self.data.dataset_1.iloc[i, attributes] if attributes else self.data.entities_d1.iloc[i]
            for token in self._tokenize_entity(record):
                index.setdefault(token, set())
                index[token].add(i if self.data.is_dirty_er else self.data.dataset_limit+i)
            self._progress_bar.update(1)
        return index
        
    def fit(
        self, data: Data, 
        attributes_1: list=None,
        attributes_2: list=None
    ) -> networkx.Graph:
        
        self.attributes_1 = attributes_1
        self.attributes_2 = attributes_2
        self.data = data
        self.pairs = networkx.Graph()
        self._progress_bar = tqdm(total=self.data.num_of_entities, desc=self._method_name+" - Processing")
        entity_index_d1 = self._create_entity_index(self.data.dataset_1, self.attributes_1)
        if not self.data.is_dirty_er:
            entity_index_d2 = self._create_entity_index(self.data.dataset_1, self.attributes_2)
        
        tokens = entity_index_d1.keys() if self.data.is_dirty_er else set(entity_index_d1.keys()).intersection(entity_index_d2.keys())
        
        self._progress_bar = tqdm(total=len(tokens), desc=self._method_name+" - Join")
        for token in tokens:
            if self.data.is_dirty_er:
                ids = list(entity_index_d1[token])
                for id1 in range(0, len(ids)):
                    for id2 in range(id1+1, len(ids)):
                        sim = self._similarity(ids[id1], ids[id2])
                        self._insert_to_graph(ids[id1], ids[id2], sim)
            else:
                ids1 = entity_index_d1[token]
                ids2 = entity_index_d2[token]
                for id1 in ids1:
                    for id2 in ids2:
                        sim = self._similarity(id1, id2)        
                        self._insert_to_graph(id1, id2, sim)
            self._progress_bar.update(1)
                        
        return self.pairs
    
    def _insert_to_graph(self, entity_id1, entity_id2, similarity):
        if self.similarity_threshold is None or \
            (self.similarity_threshold and similarity > self.similarity_threshold):
            self.pairs.add_edge(entity_id1, entity_id2, weight=similarity)
    
    def _similarity(self, entity_id1: int, entity_id2: int, attributes: any=None) -> float:

        similarity: float = 0.0

        if isinstance(attributes, dict):
            for attribute, weight in self.attributes.items():
                similarity += weight*self._metric(
                    self.data.entities.iloc[entity_id1][attribute],
                    self.data.entities.iloc[entity_id2][attribute]
                )
        if isinstance(attributes, list):
            for attribute in self.attributes:
                similarity += self._metric(
                    self.data.entities.iloc[entity_id1][attribute],
                    self.data.entities.iloc[entity_id2][attribute]
                )
                similarity /= len(self.attributes)
        else:
            # concatenated row string
            similarity = self._metric(
                self.data.entities.iloc[entity_id1].str.cat(sep=' '),
                self.data.entities.iloc[entity_id2].str.cat(sep=' ')
            )

        return similarity
            
class SchemaAgnosticJoin(AbstractJoin):
    '''
    SchemaAgnosticJoin
    '''
    
    _method_name = "Schema Agnostic Join"

    def __init__(
        self, threshold: float, metric: str, 
        tokenization: str, qgrams: int = None) -> None:
        
        super().__init__(metric, qgrams)

        self.similarity_threshold = threshold
        self.tokenization = tokenization
        self.qgrams = qgrams
        self.metric = metric
        
    
        
        
        
"""Entity Matching Prioritization Module
"""
import numpy as np
from time import time
import matplotlib.pyplot as plt
from .matching import EntityMatching
from .comparison_cleaning import (
    ComparisonPropagation,
    ProgressiveCardinalityEdgePruning,
    ProgressiveCardinalityNodePruning,
    GlobalProgressiveSortedNeighborhood,
    LocalProgressiveSortedNeighborhood,
    ProgressiveEntityScheduling)
from .vector_based_blocking import EmbeddingsNNBlockBuilding
from sklearn.metrics.pairwise import (
    cosine_similarity
)
from networkx import Graph
from py_stringmatching.similarity_measure.affine import Affine
from py_stringmatching.similarity_measure.bag_distance import BagDistance
from py_stringmatching.similarity_measure.cosine import Cosine
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.similarity_measure.editex import Editex
from py_stringmatching.similarity_measure.generalized_jaccard import \
    GeneralizedJaccard
from py_stringmatching.similarity_measure.hamming_distance import \
    HammingDistance
from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.similarity_measure.jaro import Jaro
from py_stringmatching.similarity_measure.jaro_winkler import JaroWinkler
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.monge_elkan import MongeElkan
from py_stringmatching.similarity_measure.needleman_wunsch import \
    NeedlemanWunsch
from py_stringmatching.similarity_measure.overlap_coefficient import \
    OverlapCoefficient
from py_stringmatching.similarity_measure.partial_ratio import PartialRatio
from py_stringmatching.similarity_measure.token_sort import TokenSort
from py_stringmatching.similarity_measure.partial_token_sort import \
    PartialTokenSort
from py_stringmatching.similarity_measure.ratio import Ratio
from py_stringmatching.similarity_measure.smith_waterman import SmithWaterman
from py_stringmatching.similarity_measure.soundex import Soundex
from py_stringmatching.similarity_measure.tfidf import TfIdf
from py_stringmatching.similarity_measure.tversky_index import TverskyIndex
from py_stringmatching.tokenizer.alphabetic_tokenizer import \
    AlphabeticTokenizer
from py_stringmatching.tokenizer.alphanumeric_tokenizer import \
    AlphanumericTokenizer
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.tokenizer.whitespace_tokenizer import \
    WhitespaceTokenizer
from tqdm.autonotebook import tqdm

from .evaluation import Evaluation
from .datamodel import Data, PYJEDAIFeature
from .matching import EntityMatching
from .comparison_cleaning import AbstractMetablocking
from queue import PriorityQueue
from random import sample
from .utils import sorted_enumerate

# Package import from https://anhaidgroup.github.io/py_stringmatching/v0.4.2/index.html

available_tokenizers = [
    'white_space_tokenizer', 'qgram_tokenizer', 'delimiter_tokenizer',
    'alphabetic_tokenizer', 'alphanumeric_tokenizer'
]

metrics_mapping = {
    'levenshtein' : Levenshtein(),
    'edit_distance': Levenshtein(),
    'jaro_winkler' : JaroWinkler(),
    'bag_distance' : BagDistance(),
    'editex' : Editex(),
    'cosine' : Cosine(),
    'jaro' : Jaro(),
    'soundex' : Soundex(),
    'tfidf' : TfIdf(),
    'tversky_index':TverskyIndex(),
    'ratio' : Ratio(),
    'partial_token_sort' : PartialTokenSort(),
    'partial_ratio' : PartialRatio(),
    'hamming_distance' : HammingDistance(),
    'jaccard' : Jaccard(),
    'generalized_jaccard' : GeneralizedJaccard(),
    'dice': Dice(),
    'overlap_coefficient' : OverlapCoefficient(),
    'token_sort': TokenSort(),
    'cosine_vector_similarity': cosine_similarity
}

string_metrics = [
    'bag_distance', 'editex', 'hamming_distance', 'jaro', 'jaro_winkler', 'levenshtein',
    'edit_distance', 'partial_ratio', 'partial_token_sort', 'ratio', 'soundex', 'token_sort'
]

set_metrics = [
    'cosine', 'dice', 'generalized_jaccard', 'jaccard', 'overlap_coefficient', 'tversky_index'
]

bag_metrics = [
    'tfidf'
]

vector_metrics = [
    'cosine_vector_similarity'
]

available_metrics = string_metrics + set_metrics + bag_metrics + vector_metrics

class ProgressiveMatching(EntityMatching):
    """Applies the matching process to a subset of available pairs progressively 
    """

    _method_name: str = "Progressive Matching"
    _method_info: str = "Applies the matching process to a subset of available pairs progressively "

    def __init__(
            self,
            budget: int = 0,
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)
        self._budget : int = budget

    def predict(self,
            blocks: dict,
            data: Data,
            comparison_cleaner: AbstractMetablocking = None,
            tqdm_disable: bool = False) -> Graph:
        """Main method of  progressive entity matching. Inputs a set of blocks and outputs a graph \
            that contains of the entity ids (nodes) and the similarity scores between them (edges).
            Args:
                blocks (dict): blocks of entities
                data (Data): dataset module
                tqdm_disable (bool, optional): Disables progress bar. Defaults to False.
            Returns:
                networkx.Graph: entity ids (nodes) and similarity scores between them (edges)
        """
        start_time = time()
        self.tqdm_disable = tqdm_disable
        self._comparison_cleaner: AbstractMetablocking = comparison_cleaner

        if not blocks:
            raise ValueError("Empty blocks structure")
        self.data = data
        self.pairs = Graph()
        all_blocks = list(blocks.values())
        self._progress_bar = tqdm(total=len(blocks),
                                desc=self._method_name+" ("+self.metric+")",
                                disable=self.tqdm_disable)
        if 'Block' in str(type(all_blocks[0])):
            self._predict_raw_blocks(blocks)
        elif isinstance(all_blocks[0], set):
            if(self._comparison_cleaner == None):
                raise AttributeError("No precalculated weights were given from the CC step") 
            self._predict_prunned_blocks(blocks)
        else:
            raise AttributeError("Wrong type of Blocks")
        self.execution_time = time() - start_time
        self._progress_bar.close()

        return self.pairs

    def evaluate_auc_roc(self, methods_prediction_data : list, batch_size : int = 1, proportional : bool = True) -> None:
        """For each method, takes its predictions, calculates cumulative recall and auc, plots the corresponding ROC curve
        Args:
            methods_prediction_data (list): List with each entry containing the method name and its predicted pairs
            batch_size (int, optional): Emitted pairs step at which cumulative recall is recalculated. Defaults to 1.
        Raises:
            AttributeError: No Data object
            AttributeError: No Ground Truth file
        """
        if self.data is None:
            raise AttributeError("Can not proceed to AUC ROC evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to AUC ROC evaluation without a ground-truth file. " +
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.data)
        methods_auc_roc_data = []

        for method_name, predictions in methods_prediction_data:
            cumulative_recall, normalized_auc = eval_obj.calculate_roc_auc_data(eval_obj.data, predictions, batch_size)
            methods_auc_roc_data.append((method_name, normalized_auc, cumulative_recall))

        eval_obj.visualize_roc(methods_data = methods_auc_roc_data, proportional = proportional)

    def evaluate(self,
                 prediction,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:

        if self.data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " +
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.data)
        true_positives = 0
        total_matching_pairs = prediction.number_of_edges()
        for _, (id1, id2) in self.data.ground_truth.iterrows():
            id1 = self.data._ids_mapping_1[id1]
            id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er \
                                                else self.data._ids_mapping_2[id2]
            if (id1 in prediction and id2 in prediction[id1]) or   \
                 (id2 in prediction and id1 in prediction[id2]):
                true_positives += 1

        eval_obj.calculate_scores(true_positives=true_positives,
                                  total_matching_pairs=total_matching_pairs)
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                export_to_dict,
                                with_classification_report,
                                verbose)

class HashBasedProgressiveMatching(ProgressiveMatching):
    """Applies hash based candidate graph prunning, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Hash Based Progressive Matching"
    _method_info: str = "Applies hash based candidate graph prunning, sorts retained comparisons and applies Progressive Matching"

    def __init__(
            self,
            budget: int = 0,
            w_scheme: str = 'X2',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)
        self._w_scheme : str = w_scheme

class GlobalTopPM(HashBasedProgressiveMatching):
    """Applies Progressive CEP, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Global Top Progressive Matching"
    _method_info: str = "Applies Progressive CEP, sorts retained comparisons and applies Progressive Matching"

    def __init__(
            self,
            budget: int = 0,
            w_scheme: str = 'X2',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, w_scheme, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)

    def _predict_raw_blocks(self, blocks: dict) -> None:
        pcep : ProgressiveCardinalityEdgePruning = ProgressiveCardinalityEdgePruning(self._w_scheme, self._budget)
        candidates : dict = pcep.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=None)

        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, pcep.get_precalculated_weight(entity_id, candidate_id))
         
        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True) 
        return self.pairs.edges


    def _predict_prunned_blocks(self, blocks: dict) -> None:
        pcep : ProgressiveCardinalityEdgePruning = ProgressiveCardinalityEdgePruning(self._w_scheme, self._budget)
        candidates : dict = pcep.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=self._comparison_cleaner)

        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, self._comparison_cleaner.get_precalculated_weight(entity_id, candidate_id))

        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        return self.pairs.edges

class LocalTopPM(HashBasedProgressiveMatching):
    """Applies Progressive CNP, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Global Top Progressive Matching"
    _method_info: str = "Applies Progressive CNP, sorts retained comparisons and applies Progressive Matching"

    def __init__(
            self,
            budget: int = 0,
            w_scheme: str = 'X2',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, w_scheme, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)


    def _predict_raw_blocks(self, blocks: dict) -> None:
        pcnp : ProgressiveCardinalityNodePruning = ProgressiveCardinalityNodePruning(self._w_scheme, self._budget)
        candidates : dict = pcnp.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=None)
        
        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, pcnp.get_precalculated_weight(entity_id, candidate_id))

        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True) 
        return self.pairs.edges

    def _predict_prunned_blocks(self, blocks: dict) -> None:

        pcnp : ProgressiveCardinalityNodePruning = ProgressiveCardinalityNodePruning(self._w_scheme, self._budget)
        candidates : dict = pcnp.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=self._comparison_cleaner)

        for entity_id, candidate_ids in candidates.items():
            for candidate_id in candidate_ids:
                self._insert_to_graph(entity_id, candidate_id, self._comparison_cleaner.get_precalculated_weight(entity_id, candidate_id))

        self.pairs.edges = sorted(self.pairs.edges(data=True), key=lambda x: x[2]['weight'], reverse=True) 
        return self.pairs.edges


class EmbeddingsNNBPM(ProgressiveMatching):
    """Utilizes/Creates entity embeddings, constructs neighborhoods via NN Approach and applies Progressive Matching
    """

    _method_name: str = "Embeddings NN Blocking Based Progressive Matching"
    _method_info: str = "Utilizes/Creates entity embeddings, constructs neighborhoods via NN Approach and applies Progressive Matching"

    def __init__(
            self,
            budget: int = 0,
            vectorizer: str = 'bert',
            similarity_search: str = 'faiss',
            emission: str = 'avg',
            vector_size: int = 200,
            num_of_clusters: int = 5,
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)
        self._vectorizer = vectorizer
        self._similarity_search = similarity_search
        self._vector_size = vector_size
        self._num_of_clusters = num_of_clusters
        self._emission = emission

    def _top_pair_emission(self) -> None:
        """Applies global sorting to all entity pairs produced by NN,
           and returns pairs based on distance in ascending order
        """
        self.pairs = []
        n, k = self.neighbors.shape

        for i in range(n):
            entity_id = self.ennbb._si.d1_retained_ids[i] if self.data.is_dirty_er else self.ennbb._si.d2_retained_ids[i]
            for j in range(k):
                candidate_id = self.ennbb._si.d1_retained_ids[self.neighbors[i][j]]
                self.pairs.append((entity_id, candidate_id, self.scores[i][j]))

        self.pairs = sorted(self.pairs, key=lambda x: x[2], reverse=True)
        self.pairs = [(x[0], x[1]) for x in self.pairs]

    def _avg_pair_emission(self) -> None:
        """Sorts NN neighborhoods in ascending average distance from their query entity,
           iterate over each neighborhoods' entities in ascending distance to query entity 
        """
        self.pairs = []

        average_neighborhood_distances = np.mean(self.scores, axis=1)
        sorted_neighborhoods = sorted_enumerate(average_neighborhood_distances)

        for sorted_neighborhood in sorted_neighborhoods:

            neighbor_indices = self.scores[sorted_neighborhood]
            sorted_neighbor_indices = sorted_enumerate(neighbor_indices)
            entity_id = self.ennbb._si.d1_retained_ids[sorted_neighborhood] \
            if self.data.is_dirty_er \
            else self.ennbb._si.d2_retained_ids[sorted_neighborhood]

            for neighbor_index in sorted_neighbor_indices:
                candidate_id = self.ennbb._si.d1_retained_ids[self.neighbors[sorted_neighborhood][neighbor_index]]
                self.pairs.append((candidate_id, entity_id, self.scores[sorted_neighborhood][neighbor_index]))

        self.pairs = self.pairs = [(x[0], x[1]) for x in self.pairs]

    def _produce_pairs(self):
        """Calls pairs emission based on the requested approach
        Raises:
            AttributeError: Given emission technique hasn't been defined
        """
        if(self._emission == 'avg'):
            self._avg_pair_emission()
        elif(self._emission == 'top'):
            self._top_pair_emission()
        else:
            raise AttributeError(self._emission + ' emission technique is undefined!')

    def _predict_raw_blocks(self, blocks: dict) -> None:
        self.ennbb : EmbeddingsNNBlockBuilding = EmbeddingsNNBlockBuilding(self._vectorizer, self._similarity_search)
        self.final_blocks = self.ennbb.build_blocks(data = self.data,
                     num_of_clusters = self._num_of_clusters,
                     top_k = int(max(1, int(self._budget / self.data.num_of_entities) + (self._budget % self.data.num_of_entities > 0))),
                     return_vectors = False,
                     tqdm_disable = False,
                     save_embeddings = True,
                     load_embeddings_if_exist = True,
                     with_entity_matching = False,
                     input_cleaned_blocks = blocks)

        self.scores = self.ennbb.distances
        self.neighbors = self.ennbb.neighbors
        self.final_vectors = (self.ennbb.vectors_1, self.ennbb.vectors_2)

        self._produce_pairs()
        return self.pairs

    def _predict_prunned_blocks(self, blocks: dict) -> None:
        return self._predict_raw_blocks(blocks)
    
class SimilarityBasedProgressiveMatching(ProgressiveMatching):
    """Applies similarity based candidate graph prunning, sorts retained comparisons and applies Progressive Matching
    """

    _method_name: str = "Similarity Based Progressive Matching"
    _method_info: str = "Applies similarity based candidate graph prunning, sorts retained comparisons and applies Progressive Matching"

    def __init__(
            self,
            budget: int = 0,
            pwScheme: str = 'ACF',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)
        self._pwScheme : str = pwScheme
        
        

class GlobalPSNM(SimilarityBasedProgressiveMatching):
    """Applies Global Progressive Sorted Neighborhood Matching
    """

    _method_name: str = "Global Progressive Sorted Neighborhood Matching"
    _method_info: str = "For each entity sorted accordingly to its block's tokens, " + \
                        "evaluates its neighborhood pairs defined within shifting windows of incremental size" + \
                        " and retains the globally best candidate pairs"

    def __init__(
            self,
            budget: int = 0,
            pwScheme: str = 'ACF',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, pwScheme, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)

    def _predict_raw_blocks(self, blocks: dict):
        gpsn : GlobalProgressiveSortedNeighborhood = GlobalProgressiveSortedNeighborhood(self._pwScheme, self._budget)
        candidates :  PriorityQueue = gpsn.process(blocks=blocks, data=self.data, tqdm_disable=True)
        self.pairs = []
        while(not candidates.empty()):
            _, entity_id, candidate_id = candidates.get()
            self.pairs.append((entity_id, candidate_id))
          
        return self.pairs

    def _predict_prunned_blocks(self, blocks: dict):
        raise NotImplementedError("Sorter Neighborhood Algorithms don't support prunned blocks")
    
class LocalPSNM(SimilarityBasedProgressiveMatching):
    """Applies Local Progressive Sorted Neighborhood Matching
    """

    _method_name: str = "Global Progressive Sorted Neighborhood Matching"
    _method_info: str = "For each entity sorted accordingly to its block's tokens, " + \
                        "evaluates its neighborhood pairs defined within shifting windows of incremental size" + \
                        " and retains the globally best candidate pairs"

    def __init__(
            self,
            budget: int = 0,
            pwScheme: str = 'ACF',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, pwScheme, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)

    def _predict_raw_blocks(self, blocks: dict):
        lpsn : LocalProgressiveSortedNeighborhood = LocalProgressiveSortedNeighborhood(self._pwScheme, self._budget)
        candidates :  PriorityQueue = lpsn.process(blocks=blocks, data=self.data, tqdm_disable=True)
        self.pairs = candidates 
        return self.pairs

    def _predict_prunned_blocks(self, blocks: dict):
        raise NotImplementedError("Sorter Neighborhood Algorithms don't support prunned blocks " + \
                                "(pre comparison-cleaning entities per block distribution required")
    
    

class RandomPM(ProgressiveMatching):
    """Picks a number of random comparisons equal to the available budget
    """

    _method_name: str = "Random Progressive Matching"
    _method_info: str = "Picks a number of random comparisons equal to the available budget"

    def __init__(
            self,
            budget: int = 0,
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:

        super().__init__(budget, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)

    def _predict_raw_blocks(self, blocks: dict) -> None:
        cp : ComparisonPropagation = ComparisonPropagation()
        cleaned_blocks = cp.process(blocks=blocks, data=self.data, tqdm_disable=True)
        self._predict_prunned_blocks(cleaned_blocks)

    def _predict_prunned_blocks(self, blocks: dict) -> None:
        _all_pairs = [(id1, id2) for id1 in blocks for id2 in blocks[id1]]
        _total_pairs = len(_all_pairs)
        random_pairs = sample(_all_pairs, self._budget) if self._budget <= _total_pairs else _all_pairs
        self.pairs.add_edges_from(random_pairs)
        

class PESM(HashBasedProgressiveMatching):
    """Applies Progressive Entity Scheduling Matching
    """

    _method_name: str = "Progressive Entity Scheduling Matching"
    _method_info: str = "Applies Progressive Entity Scheduling - Sorts entities in descending order of their average weight, " + \
                        "emits the top pair per entity. Finally, traverses the sorted " + \
                        "entities and emits their comparisons in descending weight order " + \
                        "within specified budget."
    def __init__(
            self,
            budget: int = 0,
            w_scheme: str = 'X2',
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = True, # unique values or not
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:
        
        super().__init__(budget, w_scheme, metric, tokenizer, similarity_threshold, qgram, tokenizer_return_set, attributes, delim_set, padding, prefix_pad, suffix_pad)


    def _predict_raw_blocks(self, blocks: dict) -> None:
        
        pes : ProgressiveEntityScheduling = ProgressiveEntityScheduling(self._w_scheme, self._budget)
        pes.process(blocks=blocks, data=self.data, tqdm_disable=True, cc=None)
        self.pairs = pes.produce_pairs()

    def _predict_prunned_blocks(self, blocks: dict):
        return self._predict_raw_blocks(blocks)
        # raise NotImplementedError("Sorter Neighborhood Algorithms doesn't support prunned blocks (lack of precalculated weights)")
        
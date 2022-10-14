"""Entity Matching Module
"""
from py_stringmatching.similarity_measure.affine import Affine
from py_stringmatching.similarity_measure.bag_distance import BagDistance
from py_stringmatching.similarity_measure.cosine import Cosine
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.similarity_measure.editex import Editex
from py_stringmatching.similarity_measure.generalized_jaccard import GeneralizedJaccard
from py_stringmatching.similarity_measure.hamming_distance import HammingDistance
from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.similarity_measure.jaro import Jaro
from py_stringmatching.similarity_measure.jaro_winkler import JaroWinkler
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.monge_elkan import MongeElkan
from py_stringmatching.similarity_measure.needleman_wunsch import NeedlemanWunsch
from py_stringmatching.similarity_measure.overlap_coefficient import OverlapCoefficient
from py_stringmatching.similarity_measure.partial_ratio import PartialRatio
from py_stringmatching.similarity_measure.partial_token_sort import PartialTokenSort
from py_stringmatching.similarity_measure.ratio import Ratio
from py_stringmatching.similarity_measure.smith_waterman import SmithWaterman
from py_stringmatching.similarity_measure.tfidf import TfIdf
from py_stringmatching.similarity_measure.soundex import Soundex
from py_stringmatching.similarity_measure.tversky_index import TverskyIndex

from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.alphabetic_tokenizer import AlphabeticTokenizer
from py_stringmatching.tokenizer.alphanumeric_tokenizer import AlphanumericTokenizer

# Package import from https://anhaidgroup.github.io/py_stringmatching/v0.4.2/index.html


from time import time
from tqdm.notebook import tqdm
from networkx import Graph
from .datamodel import Data

available_tokenizers = [
    'white_space_tokenizer', 'qgram_tokenizer', 'delimiter_tokenizer', 
    'alphabetic_tokenizer', 'alphanumeric_tokenizer'
]

available_metrics = [
    'levenshtein', 'edit_distance', 'jaro_winkler', 'bag_distance', 'editex'
    'cosine', 'soundex', 'jaro', 'partial_token_sort', 'ratio', 'partial_ratio',
    'tfidf', 'monge_elkan', 'needleman_wunsch', 'hamming_distance', 'jaccard',
    'dice', 'overlap_coefficient', 'tversky_index', 'affine'
]

class EntityMatching:
    """Calculates similarity from 0.0 to 1.0 for all blocks
    """

    _method_name: str = "Entity Matching"
    _method_info: str = "Calculates similarity from 0. to 1. for all blocks"

    def __init__(
            self,
            metric: str = 'dice',
            tokenizer: str = 'white_space_tokenizer',
            similarity_threshold: float = 0.5,
            qgram: int = 2, # for jaccard
            tokenizer_return_set = False, # unique values or not
            embedings: str = None,
            attributes: any = None,
            delim_set: list = None, # DelimiterTokenizer
            padding: bool = True, # QgramTokenizer
            prefix_pad: str = '#', # QgramTokenizer (if padding=True)
            suffix_pad: str = '$' # QgramTokenizer (if padding=True)
        ) -> None:
        self.data: Data
        self.pairs: Graph
        self.metric = metric
        self.qgram: int = qgram
        self.embedings: str = embedings
        self.attributes: list = attributes
        self.similarity_threshold = similarity_threshold
        self.tqdm_disable: bool
        self.execution_time: float
        self._progress_bar: tqdm

        # Selecting tokenizer
        if tokenizer == 'white_space_tokenizer':
            self._tokenizer = WhitespaceTokenizer(
                return_set=tokenizer_return_set)
        elif tokenizer == 'qgram_tokenizer':
            self._tokenizer = QgramTokenizer(
                return_set=tokenizer_return_set,
                padding=padding,
                suffix_pad=suffix_pad,
                prefix_pad=prefix_pad
            )
        elif tokenizer == 'delimiter_tokenizer':
            self._tokenizer = DelimiterTokenizer(
                return_set=tokenizer_return_set,
                delim_set=delim_set)
        elif tokenizer == 'alphabetic_tokenizer':
            self._tokenizer = AlphabeticTokenizer(
                return_set=tokenizer_return_set)
        elif tokenizer == 'alphanumeric_tokenizer':
            self._tokenizer = AlphanumericTokenizer(
                return_set=tokenizer_return_set)
        else:
            raise AttributeError(
                'Tokenizer ({}) does not exist. Please select one of the available. ({})'.format(
                    tokenizer, available_tokenizers
                )
            )

        # Selecting similarity measure
        if metric == 'levenshtein' or metric == 'edit_distance':
            self._metric = Levenshtein()
        elif metric == 'jaro_winkler':
            self._metric = JaroWinkler()
        elif metric == 'bag_distance':
            self._metric = BagDistance()
        elif metric == 'editex':
            self._metric = Editex()
        elif metric == 'cosine':
            self._metric = Cosine()
        elif metric == 'jaro':
            self._metric = Jaro()
        elif metric == 'soundex':
            self._metric = Soundex()
        elif metric == 'tfidf':
            self._metric = TfIdf()
        elif metric == 'tversky_index':
            self._metric = TverskyIndex()
        elif metric == 'smith_waterman':
            self._metric = SmithWaterman()
        elif metric == 'ratio':
            self._metric = Ratio()
        elif metric == 'partial_token_sort':
            self._metric = PartialTokenSort()
        elif metric == 'partial_ratio':
            self._metric = PartialRatio()
        elif metric == 'monge_elkan':
            self._metric = MongeElkan()
        elif metric == 'needleman_wunsch':
            self._metric = NeedlemanWunsch()
        elif metric == 'hamming_distance':
            self._metric = HammingDistance()
        elif metric == 'jaccard':
            self._metric = Jaccard()
        elif metric == 'generalized_jaccard':
            self._metric = GeneralizedJaccard()
        elif metric == 'dice':
            self._metric = Dice()
        elif metric == 'overlap_coefficient':
            self._metric = OverlapCoefficient()
        elif metric == 'affine':
            self._metric = Affine()
        else:
            raise AttributeError(
                'Metric ({}) does not exist. Please select one of the available. ({})'.format(
                    metric, available_metrics
                )
            )

    def predict(self,
                blocks: dict,
                data: Data,
                tqdm_disable: bool = False) -> Graph:
        """Main method of entity matching. Inputs a set of blocks and outputs a graph \
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
        if not blocks:
            raise ValueError("Empty blocks structure")
        self.data = data
        self.pairs = Graph()
        all_blocks = list(blocks.values())
        self._progress_bar = tqdm(
            total=len(blocks),
            desc=self._method_name+" ("+self.metric+")",
            disable=self.tqdm_disable
        )
        if 'Block' in str(type(all_blocks[0])):
            self._predict_raw_blocks(blocks)
        elif isinstance(all_blocks[0], set):
            self._predict_prunned_blocks(blocks)
        else:
            raise AttributeError("Wrong type of Blocks")
        self.execution_time = time() - start_time
        self._progress_bar.close()
        return self.pairs

    def _predict_raw_blocks(self, blocks: dict) -> None:
        """Method for similarity evaluation blocks after Block building

        Args:
            blocks (dict): Block building blocks
        """
        if self.data.is_dirty_er:
            for _, block in blocks.items():
                entities_array = list(block.entities_D1)
                for index_1 in range(0, len(entities_array), 1):
                    for index_2 in range(index_1+1, len(entities_array), 1):
                        similarity = self._similarity(
                            entities_array[index_1], entities_array[index_2]
                        )
                        self._insert_to_graph(
                            entities_array[index_1],
                            entities_array[index_2],
                            similarity
                        )
                self._progress_bar.update(1)
        else:
            for _, block in blocks.items():
                for entity_id1 in block.entities_D1:
                    for entity_id2 in block.entities_D2:
                        similarity = self._similarity(
                            entity_id1, entity_id2
                        )
                        self._insert_to_graph(entity_id1, entity_id2, similarity)
                self._progress_bar.update(1)

    def _predict_prunned_blocks(self, blocks: dict) -> None:
        """Similarity evaluation after comparison cleaning.

        Args:
            blocks (dict): Comparison cleaning blocks.
        """
        for entity_id, candidates in blocks.items():
            for candidate_id in candidates:
                similarity = self._similarity(entity_id, candidate_id)
                self._insert_to_graph(entity_id, candidate_id, similarity)
            self._progress_bar.update(1)

    def _insert_to_graph(self, entity_id1, entity_id2, similarity):
        if self.similarity_threshold is None or \
            (self.similarity_threshold and similarity > self.similarity_threshold):
            self.pairs.add_edge(entity_id1, entity_id2, weight=similarity)

    def _similarity(self, entity_id1: int, entity_id2: int) -> float:

        similarity: float = 0.0
        if isinstance(self.attributes, dict):
            for attribute, weight in self.attributes.items():
                similarity += weight*self._metric.get_sim_score(
                    self._tokenizer.tokenize(self.data.entities.iloc[entity_id1][attribute]),
                    self._tokenizer.tokenize(self.data.entities.iloc[entity_id2][attribute])
                )
        if isinstance(self.attributes, list):
            for attribute in self.attributes:
                similarity += self._metric.get_sim_score(
                    self._tokenizer.tokenize(self.data.entities.iloc[entity_id1][attribute]),
                    self._tokenizer.tokenize(self.data.entities.iloc[entity_id2][attribute])
                )
                similarity /= len(self.attributes)
        else:
            # concatenated row string
            similarity = self._metric.get_sim_score(
                self._tokenizer.tokenize(self.data.entities.iloc[entity_id1].str.cat(sep=' ')),
                self._tokenizer.tokenize(self.data.entities.iloc[entity_id2].str.cat(sep=' '))
            )

        return similarity

    def method_configuration(self) -> dict:
        return {
            "name" : self._method_name,
            "parameters" : self._configuration(),
            "runtime": self.execution_time
        }

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "Attributes:\n\t" + ', '.join(c for c in (self.attributes if self.attributes is not None \
                else self.data.dataset_1.columns)) +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

    def _configuration(self) -> dict:
        return {
            "Metric" : self.metric,
            "Embeddings" : self.embedings,
            "Attributes" : self.attributes,
            "Similarity threshold" : self.similarity_threshold
        }

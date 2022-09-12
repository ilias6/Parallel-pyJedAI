from typing import List
import nltk
import math
import re
import time
import logging as log
from tqdm.notebook import tqdm
import numpy as np

from .datamodel import Block, Data
from .utils import drop_big_blocks_by_size, drop_single_entity_blocks

class AbstractBlockBuilding:
    """Abstract class for the block building method
    """

    _method_name: str
    _method_info: str

    def __init__(self) -> any:
        self.blocks: dict
        self._progress_bar: tqdm
        self.attributes_1: list
        self.attributes_2: list
        self.execution_time: float
        self.data: Data

    def build_blocks(
            self,
            data: Data,
            attributes_1: list = None,
            attributes_2: list = None,
            tqdm_disable: bool = False
    ) -> dict:
        """Main method of Blocking in a dataset

        Args:
            data (Data): Data module that contaiins the processed dataset
            attributes_1 (list, optional): Attribute columns of the dataset 1 \
                that will be processed. Defaults to None. \
                If not provided, all attributes are slected.
            attributes_2 (list, optional): Attribute columns of the dataset 2. \
                Defaults to None. If not provided, all attributes are slected.
            tqdm_disable (bool, optional): Disables all tqdm at processing. Defaults to False.

        Returns:
            dict: Blocks as a dictionary of keys to sets of Block objects (Block contains two sets).
        """

        _start_time = time.time()
        self.blocks = dict()
        self.data, self.attributes_1, self.attributes_2 = data, attributes_1, attributes_2
        self._progress_bar = tqdm(
            total=data.num_of_entities, desc=self._method_name, disable=tqdm_disable
        )

        if attributes_1:
            isolated_attr_dataset_1 = data.dataset_1[attributes_1].apply(" ".join, axis=1)
        if attributes_2:
            isolated_attr_dataset_2 = data.dataset_2[attributes_1].apply(" ".join, axis=1)

        for i in range(0, data.num_of_entities_1, 1):
            record = isolated_attr_dataset_1.iloc[i] if attributes_1 \
                        else data.entities_d1.iloc[i]
            for token in self._tokenize_entity(record):
                self.blocks.setdefault(token, Block())
                self.blocks[token].entities_D1.add(i)
            self._progress_bar.update(1)
        if not data.is_dirty_er:
            for i in range(0, data.num_of_entities_2, 1):
                record = isolated_attr_dataset_2.iloc[i] if attributes_2 \
                            else data.entities_d2.iloc[i]
                for token in self._tokenize_entity(record):
                    self.blocks.setdefault(token, Block())
                    self.blocks[token].entities_D2.add(data.dataset_limit+i)
                self._progress_bar.update(1)

        self.blocks = self._clean_blocks(drop_single_entity_blocks(self.blocks, data.is_dirty_er))
        self.execution_time = time.time() - _start_time
        self._progress_bar.close()
        return self.blocks

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
            "Attributes from D1:\n\t" + ', '.join(c for c in (self.attributes_1 if self.attributes_1 is not None \
                else self.data.dataset_1.columns)) +
            ("\nAttributes from D2:\n\t" + ', '.join(c for c in (self.attributes_2 if self.attributes_2 is not None \
                else self.data.dataset_2.columns)) if not self.data.is_dirty_er else "") +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

class StandardBlocking(AbstractBlockBuilding):
    """ Creates one block for every token in \
        the attribute values of at least two entities.
    """

    _method_name = "Standard Blocking"
    _method_info = "Creates one block for every token in the attribute values of at least two entities."

    def __init__(self) -> any:
        super().__init__()

    def _tokenize_entity(self, entity: str) -> set:
        """Produces a list of workds of a given string

        Args:
            entity (str): String representation  of an entity

        Returns:
            list: List of words
        """
        return set(filter(None, re.split('[\\W_]', entity.lower())))

    def _clean_blocks(self, blocks: dict) -> dict:
        """No cleaning"""
        return blocks

    def _configuration(self) -> dict:
        """No configuration"""
        return {}

class QGramsBlocking(StandardBlocking):
    """ Creates one block for every q-gram that is extracted \
        from any token in the attribute values of any entity. \
            The q-gram must be shared by at least two entities.
    """

    _method_name = "Q-Grams Blocking"
    _method_info = "Creates one block for every q-gram that is extracted " + \
                    "from any token in the attribute values of any entity. " + \
                    "The q-gram must be shared by at least two entities."

    def __init__(
            self, qgrams: int = 6
    ) -> any:
        super().__init__()
        self.qgrams = qgrams

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        for token in super()._tokenize_entity(entity):
            if len(token) < self.qgrams:
                keys.add(token)
            else:
                keys.update(''.join(qg) for qg in nltk.ngrams(token, n=self.qgrams))
        return keys

    def _clean_blocks(self, blocks: dict) -> dict:
        return blocks

    def _configuration(self) -> dict:
        return {
            "Q-Gramms" : self.qgrams
        }

class SuffixArraysBlocking(StandardBlocking):
    """ It creates one block for every suffix that appears \
        in the attribute value tokens of at least two entities.
    """

    _method_name = "Suffix Arrays Blocking"
    _method_info = "Creates one block for every suffix that appears in the " + \
        "attribute value tokens of at least two entities."

    def __init__(
            self,
            suffix_length: int = 6,
            max_block_size: int = 53
    ) -> any:
        super().__init__()
        self.suffix_length, self.max_block_size = suffix_length, max_block_size

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        for token in super()._tokenize_entity(entity):
            if len(token) < self.suffix_length:
                keys.add(token)
            else:
                for length in range(0, len(token) - self.suffix_length + 1):
                    keys.add(token[length:])
        return keys

    def _clean_blocks(self, blocks: dict) -> dict:
        return drop_big_blocks_by_size(blocks, self.max_block_size)

    def _configuration(self) -> dict:
        return {
            "Suffix length" : self.suffix_length,
            "Maximum Block Size" : self.max_block_size
        }

class ExtendedSuffixArraysBlocking(StandardBlocking):
    """ It creates one block for every substring \
        (not just suffix) that appears in the tokens of at least two entities.
    """

    _method_name = "Extended Suffix Arrays Blocking"
    _method_info = "Creates one block for every substring (not just suffix) " + \
        "that appears in the tokens of at least two entities."

    def __init__(
            self,
            suffix_length: int = 6,
            max_block_size: int = 39
    ) -> any:
        super().__init__()
        self.suffix_length, self.max_block_size = suffix_length, max_block_size

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        for token in super()._tokenize_entity(entity):
            keys.add(token)
            if len(token) > self.suffix_length:
                for current_size in range(self.suffix_length, len(token)): 
                    for letters in list(nltk.ngrams(token, n=current_size)):
                        keys.add("".join(letters))
        return keys

    def _clean_blocks(self, blocks: dict) -> dict:
        return drop_big_blocks_by_size(blocks, self.max_block_size)

    def _configuration(self) -> dict:
        return {
            "Suffix length" : self.suffix_length,
            "Maximum Block Size" : self.max_block_size
        }

class ExtendedQGramsBlocking(StandardBlocking):
    """It creates one block for every combination of q-grams that represents at least two entities.
    The q-grams are extracted from any token in the attribute values of any entity.
    """

    _method_name = "Extended QGramsBlocking"
    _method_info = "Creates one block for every substring (not just suffix) " + \
        "that appears in the tokens of at least two entities."

    def __init__(
            self,
            qgrams: int = 6,
            threshold: float = 0.95
    ) -> any:
        super().__init__()
        self.threshold: float = threshold
        self.MAX_QGRAMS: int = 15
        self.qgrams = qgrams

    def _tokenize_entity(self, entity) -> set:
        keys = set()
        for token in super()._tokenize_entity(entity):
            if len(token) < self.qgrams:
                keys.add(token)
            else:   
                qgrams = [''.join(qgram) for qgram in nltk.ngrams(token, n=self.qgrams)]
                if len(qgrams) == 1:
                    keys.update(qgrams)
                else:
                    if len(qgrams) > self.MAX_QGRAMS:
                        qgrams = qgrams[:self.MAX_QGRAMS]

                    minimum_length = max(1, math.floor(len(qgrams) * self.threshold))
                    for i in range(minimum_length, len(qgrams) + 1):
                        keys.update(self._qgrams_combinations(qgrams, i))

        return keys

    def _qgrams_combinations(self, sublists: list, sublist_length: int) -> list:
        if sublist_length == 0 or len(sublists) < sublist_length:
            return []

        remaining_elements = sublists.copy()
        last_sublist = remaining_elements.pop(len(sublists)-1)

        combinations_exclusive_x = self._qgrams_combinations(remaining_elements, sublist_length)
        combinations_inclusive_x = self._qgrams_combinations(remaining_elements, sublist_length-1)

        resulting_combinations = combinations_exclusive_x.copy() if combinations_exclusive_x else []

        if not combinations_inclusive_x: # is empty
            resulting_combinations.append(last_sublist)
        else:
            for combination in combinations_inclusive_x:
                resulting_combinations.append(combination+last_sublist)

        return resulting_combinations

    def _clean_blocks(self, blocks: dict) -> dict:
        return blocks

    def _configuration(self) -> dict:
        return {
            "Q-Gramms" : self.qgrams,
            "Threshold" : self.threshold
        }

from gensim.models.fasttext import load_facebook_model
import gensim.downloader as api
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel
from sentence_transformers import SentenceTransformer
from transformers import AlbertTokenizer, AlbertModel
import transformers
transformers.logging.set_verbosity_error()
import torch
import faiss

class EmbeddingsNNBlockBuilding(StandardBlocking):
    """Block building via creation of embeddings and a Nearest Neighbor Approach.
    """

    _method_name = "Embeddings-NN Block Building"
    _method_info = "Creates a set of candidate pais for every entity id " + \
        "based on Embeddings creariot and Similarity search among the vectors."

    _gensim_mapping_download = {
        'fasttext' : 'fasttext-wiki-news-subwords-300',
        'glove' : 'glove-wiki-gigaword-300',
        'word2vec' : 'word2vec-google-news-300'
    }

    def __init__(
            self,
            vectorizer: str,
            similarity_search: str
    ) -> None:
        self.vectorizer, self.similarity_search = vectorizer, similarity_search
        self.embeddings: np.array
        self.vectors_1: np.array
        self.vectors_2: np.array = None
        self.vector_size: int
        self.num_of_clusters: int
        self.top_k: int

    def build_blocks(
            self,
            data: Data,
            vector_size: int = 300,
            num_of_clusters: int = 5,
            top_k: int = 30,
            attributes_1: list = None,
            attributes_2: list = None,
            tqdm_disable: bool = False
    ) -> dict:
        """
        TODO
        """
        _start_time = time.time()
        self.blocks = dict()
        self.data, self.attributes_1, self.attributes_2, self.vector_size, self.num_of_clusters, self.top_k \
            = data, attributes_1, attributes_2, vector_size, num_of_clusters, top_k
        self._progress_bar = tqdm(
            total=data.num_of_entities, desc=self._method_name, disable=tqdm_disable
        )
        if attributes_1:
            isolated_attr_dataset_1 = data.dataset_1[attributes_1].apply(" ".join, axis=1)
        if attributes_2:
            isolated_attr_dataset_2 = data.dataset_2[attributes_1].apply(" ".join, axis=1)

        vectors_1 = []

        if self.vectorizer in ['word2vec', 'fasttext', 'doc2vec', 'glove']:
            """More pre-trained embeddings: https://github.com/RaRe-Technologies/gensim-data
            """
            vocabulary = api.load(self._gensim_mapping_download[self.vectorizer])
            for i in range(0, data.num_of_entities_1, 1):
                record = isolated_attr_dataset_1.iloc[i] if attributes_1 \
                            else data.entities_d1.iloc[i]
                vectors_1.append(
                    self._create_vector(self._tokenize_entity(record), vocabulary)
                )
                self._progress_bar.update(1)
            self.vectors_1 = np.array(vectors_1).astype('float32')
            if not data.is_dirty_er:
                vectors_2 = []
                for i in range(0, data.num_of_entities_2, 1):
                    record = isolated_attr_dataset_2.iloc[i] if attributes_2 \
                                else data.entities_d2.iloc[i]
                    vectors_2.append(
                        self._create_vector(self._tokenize_entity(record), vocabulary)
                    )
                    self._progress_bar.update(1)
                self.vectors_2 = np.array(vectors_2).astype('float32')
        elif self.vectorizer in ['bert', 'distilbert', 'roberta', 'xlnet', 'albert']:
            if self.vectorizer == 'bert':
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model = BertModel.from_pretrained("bert-base-uncased")
            elif self.vectorizer == 'distilbert':
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            elif self.vectorizer == 'roberta':
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                model = RobertaModel.from_pretrained('roberta-base')
            elif self.vectorizer == 'xlnet':
                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
                model = XLNetModel.from_pretrained('xlnet-base-cased')
            elif self.vectorizer == 'albert':
                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                model = AlbertModel.from_pretrained("albert-base-v2")
                
            for i in range(0, 100, 1):
                record = isolated_attr_dataset_1.iloc[i] if attributes_1 \
                            else data.entities_d1.iloc[i]
                encoded_input = tokenizer(
                    record,
                    return_tensors='pt',
                    truncation=True,
                    max_length=100,
                    padding='max_length'
                )
                output = model(**encoded_input)
                vector = output.last_hidden_state[:, 0, :]
                vectors_1.append(vector.detach().numpy().astype('float32'))
                self._progress_bar.update(1)
            self.vectors_1 = vectors_1
            if not data.is_dirty_er:
                vectors_2 = []
                for i in range(0, 100, 1):
                    record = isolated_attr_dataset_2.iloc[i] if attributes_2 \
                                else data.entities_d2.iloc[i]
                    encoded_input = tokenizer(
                        record,
                        return_tensors='pt',
                        truncation=True,
                        max_length=100,
                        padding='max_length'
                    )
                    output = model(**encoded_input)
                    vector = output.last_hidden_state[:, 0, :]
                    vectors_2.append(vector)
                    self._progress_bar.update(1)
                self.vectors_2 = np.array(vectors_2).astype('float32')
        else:
            raise AttributeError("Not available vectorizer")

        if self.similarity_search == 'faiss':
            quantiser = faiss.IndexFlatL2(self.vectors_1.shape[1])
            index = faiss.IndexIVFFlat(
                quantiser,
                self.vectors_1.shape[1],
                self.num_of_clusters,
                faiss.METRIC_L2
            )
            index.train(self.vectors_1)  # train on the vectors of dataset 1
            index.add(self.vectors_1)   # add the vectors and update the index
            _, indices = index.search(self.vectors_1 if self.data.is_dirty_er else self.vectors_2, self.top_k)
            if self.data.is_dirty_er:
                self.blocks = {
                    i : set(x for x in indices[i] if x not in [-1, i]) \
                            for i in range(0, indices.shape[0])
                }
            else:
                self.blocks = {
                    i+self.data.dataset_limit : set(x for x in indices[i] if x != -1) \
                            for i in range(0, indices.shape[0])
                }

        # TODO elif similarity_search == 'falconn':
        # TODO elif similarity_search == 'scann':
        else:
            raise AttributeError("Not available method")
        self._progress_bar.close()
        self.execution_time = time.time() - _start_time
        return self.blocks

    def _create_vector(self, tokens: List[str], vocabulary) -> np.array:
        num_of_tokens = 0
        vector = np.zeros(self.vector_size)
        for token in tokens:
            if token in vocabulary:
                vector += vocabulary[token]
                num_of_tokens += 1
        if num_of_tokens > 0:
            vector /= num_of_tokens
        return vector

    def _configuration(self) -> dict:
        return {
            "Vectorizer" : self.vectorizer,
            "Similarity-Search" : self.similarity_search,
            "Top-K" : self.top_k,
            "Vector size": self.vector_size
        }

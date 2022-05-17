import logging
import os
import sys
from typing import Dict, List

info = logging.info
error = logging.error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.tokenizer import Tokenizer
from src.core.entities import AttributeClusters

class AbstractBlockBuilding:
    '''
    Abstract class for the block building method
    '''

    blocks_dict: dict = dict()

    



    schema_clusters: List[AttributeClusters]

    def __init__(self) -> any:
        self.is_using_cross_entropy: bool = False


    def __str__(self) -> str:
        pass


class StandardBlocking(AbstractBlockBuilding):

    _method_name = "Standard Blocking"
    _method_info = _method_name + ": it creates one block for every token in the attribute \
                                    values of at least two entities."

    def __init__(self) -> any:
        super().__init__()

class StandardBlocking(AbstractBlockBuilding):
    pass

class QGramsBlocking(AbstractBlockBuilding):
    pass

class SuffixArraysBlocking(AbstractBlockBuilding):
    pass

class LSHSuperBitBlocking(AbstractBlockBuilding):
    pass


class LSHMinHashBlocking(LSHSuperBitBlocking):
    pass


    

    
from html import entities
import logging
from typing import Dict

class Block:

    _key: any
    _is_dirty_er: bool = False
    _comparisons: int = 0
    _entropy: float = 1.0
    _cardinality: int = 0

    entities_D1: set = set()
    entities_D2: set = set()

    def __init__(self, key):
        self.key = key

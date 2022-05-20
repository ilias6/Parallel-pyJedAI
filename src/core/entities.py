from html import entities
import logging
from typing import Dict

class Block:

    _key: any
    _is_dirty_er: bool = False
    _comparisons: int = 0
    _entropy: float = 1.0
    _cardinality: int = 0

    def __init__(self, key, is_dirty_er: bool = True):
        self.key = key
        self.entities_D1: set = set()

        if not is_dirty_er:
            self._is_dirty_er = False
            self.entities_D2: set = set()

import colorama
from colorama import Fore
import logging
from typing import Dict

class Block:
    '''
    Block entity
    ---
    Consists of 2 sets of profile entities (1 for Dirty ER and 2 for Clean-Clean ER)
    '''

    def __init__(self, key, is_dirty_er: bool = True):
        self.key = key
        self.entities_D1: set = set()
        if not is_dirty_er:
            self.entities_D2: set = set()
        self._is_dirty_er = is_dirty_er

    def get_cardinality(self) -> int:
        if self._is_dirty_er:
            return len(self.entities_D1)
        return len(self.entities_D1) * len(self.entities_D2)

    def is_dirty_er(self):
        return self._is_dirty_er

    def verbose(self):
        print("\nBlock ", "\033[1;32m"+self.key+"\033[0m", " contains entities with ids: ")
        if self._is_dirty_er:
            print("Dirty dataset: " + "[\033[1;34m"+str(len(self.entities_D1)) + " entities\033[0m]")
            print(self.entities_D1)
        else:
            print("Clean dataset 1: " + "[\033[1;34m"+str(len(self.entities_D1)) + " entities\033[0m]")
            print(self.entities_D1)
            print("Clean dataset 2: " + "[\033[1;34m" + str(len(self.entities_D2)) + " entities\033[0m]")
            print(self.entities_D2)
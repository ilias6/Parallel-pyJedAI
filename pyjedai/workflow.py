import pandas as pd
from time import time
from tqdm.notebook import tqdm
from networkx import Graph
from .datamodel import Data
from .evaluation import Evaluation

class WorkFlow:

    def __init__(
            self,
            block_building: dict,
            entity_matching: dict = None,
            block_cleaning: dict = None,
            comparison_cleaning: dict = None,
            clustering: dict = None,
            joins: dict = None
    ) -> None:
        self.block_cleaning, self.block_building, self.comparison_cleaning, self.clustering, self.joins, self.entity_matching = \
            block_cleaning, block_building, comparison_cleaning, clustering, joins, entity_matching
        # self.f1

    def run(self, data: Data, verbose=False, tqdm_disable=False) -> pd.DataFrame:
        """Main function for creating an Entity resolution workflow.

        Args:
            data (Data): _description_
            verbose (bool, optional): _description_. Defaults to False.
            tqdm_disable (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        pj_eval = Evaluation(data)
        #
        # Block building step: Only one algorithm can be performed
        #
        block_building_method = self.block_building['method'](**self.block_building["params"])
        block_building_blocks = block_building_method.build_blocks(
            data,
            attributes_1=self.block_building["attributes_1"] \
                            if "attributes_1" in self.block_building else None,
            attributes_2=self.block_building["attributes_2"] \
                            if "attributes_2" in self.block_building else None,
            tqdm_disable=tqdm_disable
        )
        pj_eval.report(block_building_blocks, block_building_method.method_configuration())
        #
        # Block cleaning step [optional]: Multiple algorithms
        #
        if self.block_cleaning:
            if isinstance(self.block_cleaning, dict):
                self.block_cleaning = list(self.block_cleaning)
            bblocks = block_building_blocks
            for bc in self.block_cleaning:
                block_cleaning_method = bc['method'](**bc["params"])
                block_cleaning_blocks = block_cleaning_method.process(
                    bblocks, data, tqdm_disable=tqdm_disable
                )
                bblocks = block_cleaning_blocks
                pj_eval.report(bblocks, block_cleaning_method.method_configuration())
        #
        # Comparison cleaning step [optional]
        #
        if self.comparison_cleaning:
            comparison_cleaning_method = self.comparison_cleaning['method'](**self.comparison_cleaning["params"]) \
                                            if "params" in self.comparison_cleaning \
                                            else self.comparison_cleaning['method']()
            comparison_cleaning_blocks = comparison_cleaning_method.process(
                block_cleaning_blocks if block_cleaning_blocks is not None \
                    else block_building_blocks,
                data,
                tqdm_disable=tqdm_disable
            )
            pj_eval.report(comparison_cleaning_blocks, comparison_cleaning_method.method_configuration())
        #
        # Entity Matching step
        #
        entity_matching_method = self.entity_matching['method'](**self.entity_matching["params"]) \
                                        if "params" in self.entity_matching \
                                        else self.entity_matching['method']()
        em_graph = entity_matching_method.predict(
            comparison_cleaning_blocks if comparison_cleaning_blocks is not None \
                else block_building_blocks,
            data,
            tqdm_disable=tqdm_disable
        )
        pj_eval.report(em_graph, entity_matching_method.method_configuration())
        #
        # Clustering step [optional]
        #
        clustering_method = self.clustering['method'](**self.clustering["params"]) \
                                        if "params" in self.entity_matching \
                                        else self.clustering['method']()
        components = clustering_method.process(em_graph)
        pj_eval.report(components, clustering_method.method_configuration())
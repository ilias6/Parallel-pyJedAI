from turtle import color
from typing import List
import pandas as pd
from time import time
import matplotlib.pyplot as plt
# import plotly.express as px
from tqdm.notebook import tqdm
from itertools import count
from networkx import Graph
from .datamodel import Data
from .evaluation import Evaluation

plt.style.use('seaborn-whitegrid')

class WorkFlow:
    """Main module of the pyjedAI and the simplest way to create an end-to-end ER workflow.
    """

    _id = count()

    def __init__(
            self,
            block_building: dict,
            entity_matching: dict = None,
            block_cleaning: dict = None,
            comparison_cleaning: dict = None,
            clustering: dict = None,
            joins: dict = None,
            name: str = None
    ) -> None:
        self.block_cleaning, self.block_building, self.comparison_cleaning, \
            self.clustering, self.joins, self.entity_matching = \
            block_cleaning, block_building, comparison_cleaning, clustering, joins, entity_matching
        self.f1: list = []
        self.recall: list = []
        self.precision: list = []
        self.runtime: list = []
        self.configurations: list = []
        self.workflow_exec_time: float
        self._id: int = next(self._id)
        self.name: str = name if name else "Workflow-" + str(self._id)

    def run(self, data: Data, verbose=False, tqdm_disable=False) -> pd.DataFrame:
        """Main function for creating an Entity resolution workflow.

        Args:
            data (Data): _description_
            verbose (bool, optional): _description_. Defaults to False.
            tqdm_disable (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        self._init_experiment()
        start_time = time()
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
        pj_eval.report(
            block_building_blocks, block_building_method.method_configuration(), verbose=verbose
        )
        self._save_step(pj_eval, block_building_method.method_configuration())
        #
        # Block cleaning step [optional]: Multiple algorithms
        #
        if self.block_cleaning:
            if isinstance(self.block_cleaning, dict):
                self.block_cleaning = list(self.block_cleaning)
            bblocks = block_building_blocks
            for block_cleaning in self.block_cleaning:
                block_cleaning_method = block_cleaning['method'](**block_cleaning["params"])
                block_cleaning_blocks = block_cleaning_method.process(
                    bblocks, data, tqdm_disable=tqdm_disable
                )
                bblocks = block_cleaning_blocks
                pj_eval.report(
                    bblocks, block_cleaning_method.method_configuration(), verbose=verbose
                )
                self._save_step(pj_eval, block_cleaning_method.method_configuration())
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
            pj_eval.report(
                comparison_cleaning_blocks,
                comparison_cleaning_method.method_configuration(),
                verbose=verbose
            )
            self._save_step(pj_eval, comparison_cleaning_method.method_configuration())
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
        pj_eval.report(
            em_graph, entity_matching_method.method_configuration(), verbose=verbose
        )
        self._save_step(pj_eval, entity_matching_method.method_configuration())
        #
        # Clustering step [optional]
        #
        clustering_method = self.clustering['method'](**self.clustering["params"]) \
                                        if "params" in self.entity_matching \
                                        else self.clustering['method']()
        components = clustering_method.process(em_graph)
        pj_eval.report(
            components, clustering_method.method_configuration(), verbose=verbose
        )
        self._save_step(pj_eval, clustering_method.method_configuration())
        self.workflow_exec_time = time() - start_time
        # self.runtime.append(self.workflow_exec_time)

    def _init_experiment(self) -> None:
        self.f1: list = []
        self.recall: list = []
        self.precision: list = []
        self.runtime: list = []
        self.configurations: list = []
        self.workflow_exec_time: float

    def visualize(
            self,
            f1: bool = True,
            recall: bool = True,
            precision: bool = True,
            # runtime:bool = True,
            separate: bool = False
    ):
        method_names = [conf['name'] for conf in self.configurations]
        if separate:
            fig, axs = plt.subplots(2, 2)
        else:
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
            fig.suptitle(self.name + " Visualization", fontweight='bold', fontsize=14)
            fig.subplots_adjust(top=0.88)
            if precision: axs[0].plot(method_names, self.precision, linewidth=2.0, label="Precision", marker='o', markersize=10)
            if recall: axs[0].plot(method_names, self.recall, linewidth=2.0, label="Recall", marker='*', markersize=10)
            if f1: axs[0].plot(method_names, self.f1, linewidth=2.0, label="F1-Score", marker='x', markersize=10)
            axs[0].set_xlabel("Models", fontsize=12)
            axs[0].set_ylabel("Scores %", fontsize=12)
            axs[0].set_title("Performance per step", fontsize=12)
            axs[0].legend(loc='lower right')
            exec_time = []
            prev = 0
            for i in range(0, len(self.runtime)):
                exec_time.append(prev + self.runtime[i])
                prev = exec_time[i]
            axs[1].plot(method_names, exec_time, linewidth=2.0, label="F1-Score", marker='.', markersize=10, color='r')
            axs[1].set_ylabel("Time (sec)", fontsize=12)
            axs[1].set_title("Execution time", fontsize=12)
            fig.autofmt_xdate()
        plt.show()

    def to_df(self) -> pd.DataFrame:
        workflow_df = pd.DataFrame(columns=['Algorithm', 'F1', 'Recall', 'Precision', 'Runtime (sec)', 'Params'])
        workflow_df['F1'], workflow_df['Recall'], workflow_df['Precision'], workflow_df['Runtime (sec)'] = \
            self.f1, self.recall, self.precision, self.runtime
        workflow_df['Algorithm'] = [c['name'] for c in self.configurations]
        workflow_df['Params'] = [c['parameters'] for c in self.configurations]
        return workflow_df

    def _save_step(self, evaluation: Evaluation, configuration: dict) -> None:
        self.f1.append(evaluation.f1*100)
        self.recall.append(evaluation.recall*100)
        self.precision.append(evaluation.precision*100)
        self.configurations.append(configuration)
        self.runtime.append(configuration['runtime'])

# def compare_workflows(workflows: List[WorkFlow], with_visualization=True) -> pd.DataFrame:
#     for w in workflows:
        
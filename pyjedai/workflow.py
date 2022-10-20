from itertools import count
from time import time
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from networkx import Graph
# import plotly.express as px
from tqdm.autonotebook import tqdm

from .datamodel import Data
from .evaluation import Evaluation, write

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
        self._workflow_bar: tqdm
        self.final_pairs = None

    def run(
            self,
            data: Data,
            verbose=False,
            workflow_step_tqdm_disable=True,
            workflow_tqdm_enable=False
        ) -> None:
        """Main function for creating an Entity resolution workflow.

        Args:
            data (Data): Dataset module.
            verbose (bool, optional): Print detailed report for each step. Defaults to False.
            workflow_step_tqdm_disable (bool, optional): Tqdm progress bar. Defaults to False.
        """
        steps = [self.block_building, self.entity_matching, self.clustering, self.joins, self.block_cleaning, self.comparison_cleaning]
        num_of_steps = sum(x is not None for x in steps)
        self._workflow_bar = tqdm(total=num_of_steps,
                                  desc=self.name,
                                  disable=not workflow_tqdm_enable)
        self.data = data
        self._init_experiment()
        start_time = time()
        pj_eval = Evaluation(data)
        #
        # Block building step: Only one algorithm can be performed
        #
        block_building_method = self.block_building['method'](**self.block_building["params"]) \
                                                    if "params" in self.block_building \
                                                    else self.block_building['method']()
        block_building_blocks, entity_index  = \
        block_building_method.build_blocks(data,
                                            attributes_1=self.block_building["attributes_1"] \
                                                            if "attributes_1" in self.block_building else None,
                                            attributes_2=self.block_building["attributes_2"] \
                                                            if "attributes_2" in self.block_building else None,
                                            tqdm_disable=workflow_step_tqdm_disable)
        self.final_pairs = block_building_blocks
        pj_eval.report(block_building_blocks, 
                       block_building_method.method_configuration(), 
                       verbose=verbose)
        self._save_step(pj_eval, block_building_method.method_configuration())
        self._workflow_bar.update(1)
        #
        # Block cleaning step [optional]: Multiple algorithms
        #
        block_cleaning_blocks = None
        if self.block_cleaning:
            if isinstance(self.block_cleaning, dict):
                self.block_cleaning = list(self.block_cleaning)
            bblocks = block_building_blocks
            for block_cleaning in self.block_cleaning:
                block_cleaning_method = block_cleaning['method'](**block_cleaning["params"]) \
                                                    if "params" in block_cleaning \
                                                    else block_cleaning['method']()
                block_cleaning_blocks, entity_index = block_cleaning_method.process(bblocks, 
                                                                      data, 
                                                                      tqdm_disable=workflow_step_tqdm_disable)
                
                self.final_pairs = bblocks = block_cleaning_blocks
                pj_eval.report(
                    bblocks, block_cleaning_method.method_configuration(), verbose=verbose
                )
                self._save_step(pj_eval, block_cleaning_method.method_configuration())
                self._workflow_bar.update(1)
        #
        # Comparison cleaning step [optional]
        #
        comparison_cleaning_blocks = None
        if self.comparison_cleaning:
            comparison_cleaning_method = self.comparison_cleaning['method'](**self.comparison_cleaning["params"]) \
                                            if "params" in self.comparison_cleaning \
                                            else self.comparison_cleaning['method']()
            self.final_pairs = \
            comparison_cleaning_blocks = \
            comparison_cleaning_method.process(block_cleaning_blocks if block_cleaning_blocks is not None \
                                                    else block_building_blocks,
                                                data,
                                                entity_index=entity_index,
                                                tqdm_disable=workflow_step_tqdm_disable)
            pj_eval.report(comparison_cleaning_blocks,
                           comparison_cleaning_method.method_configuration(),
                           verbose=verbose)
            self._save_step(pj_eval, comparison_cleaning_method.method_configuration())
            self._workflow_bar.update(1)
        #
        # Entity Matching step
        #
        entity_matching_method = self.entity_matching['method'](**self.entity_matching["params"]) \
                                        if "params" in self.entity_matching \
                                        else self.entity_matching['method']()
        self.final_pairs = em_graph = entity_matching_method.predict(
            comparison_cleaning_blocks if comparison_cleaning_blocks is not None \
                else block_building_blocks,
            data,
            tqdm_disable=workflow_step_tqdm_disable
        )
        pj_eval.report(
            em_graph, entity_matching_method.method_configuration(), verbose=verbose
        )
        self._save_step(pj_eval, entity_matching_method.method_configuration())
        self._workflow_bar.update(1)
        #
        # Clustering step [optional]
        #
        if self.clustering:
            clustering_method = self.clustering['method'](**self.clustering["params"]) \
                                            if "params" in self.entity_matching \
                                            else self.clustering['method']()
            self.final_pairs = components = clustering_method.process(em_graph)
            pj_eval.report(
                components, clustering_method.method_configuration(), verbose=verbose
            )
            self._save_step(pj_eval, clustering_method.method_configuration())
            self.workflow_exec_time = time() - start_time
            self._workflow_bar.update(1)
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
            separate: bool = False
    ) -> None:
        method_names = [conf['name'] for conf in self.configurations]
        exec_time = []
        prev = 0
        for i in range(0, len(self.runtime)):
            exec_time.append(prev + self.runtime[i])
            prev = exec_time[i]
        if separate:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
            fig.suptitle(self.name + " Visualization", fontweight='bold', fontsize=14)
            fig.subplots_adjust(top=0.88)
            axs[0, 0].plot(method_names, self.precision, linewidth=2.0, label="Precision", marker='o', markersize=10)
            axs[0, 0].set_ylabel("Scores %", fontsize=12)
            axs[0, 0].set_title("Precision", fontsize=12)
            axs[0, 1].plot(method_names, self.recall, linewidth=2.0, label="Recall", marker='*', markersize=10)
            axs[0, 1].set_ylabel("Scores %", fontsize=12)
            axs[0, 1].set_title("Recall", fontsize=12)            
            axs[1, 0].plot(method_names, self.f1, linewidth=2.0, label="F1-Score", marker='x', markersize=10)
            axs[1, 0].set_ylabel("Scores %", fontsize=12)
            axs[1, 0].set_title("F1-Score", fontsize=12)
            # axs[0, 0].legend(loc='lower right')
            axs[1, 1].plot(method_names, exec_time, linewidth=2.0, label="Time", marker='.', markersize=10, color='r')
            axs[1, 1].set_ylabel("Time (sec)", fontsize=12)
            axs[1, 1].set_title("Execution time", fontsize=12)
            fig.autofmt_xdate()
        else:
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
            fig.suptitle(self.name + " Visualization", fontweight='bold', fontsize=14)
            fig.subplots_adjust(top=0.88)
            if precision:
                axs[0].plot(method_names, self.precision, linewidth=2.0, label="Precision", marker='o', markersize=10)
            if recall:
                axs[0].plot(method_names, self.recall, linewidth=2.0, label="Recall", marker='*', markersize=10)
            if f1:
                axs[0].plot(method_names, self.f1, linewidth=2.0, label="F1-Score", marker='x', markersize=10)
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

    def export_pairs(self) -> pd.DataFrame:
        return write(self.final_pairs, self.data)

    def _save_step(self, evaluation: Evaluation, configuration: dict) -> None:
        self.f1.append(evaluation.f1*100)
        self.recall.append(evaluation.recall*100)
        self.precision.append(evaluation.precision*100)
        self.configurations.append(configuration)
        self.runtime.append(configuration['runtime'])
        
    def get_final_scores(self) -> Tuple[float, float, float]:
        return self.f1[-1], self.precision[-1], self.recall[-1]

def compare_workflows(workflows: List[WorkFlow], with_visualization=True) -> pd.DataFrame:
    workflow_df = pd.DataFrame(columns=['Name', 'F1', 'Recall', 'Precision', 'Runtime (sec)'])
    if with_visualization:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        fig.suptitle("Workflows Performance Visualization", fontweight='bold', fontsize=14)
        fig.subplots_adjust(top=0.88)
        axs[0, 0].set_ylabel("Scores %", fontsize=12)
        axs[0, 0].set_title("Precision", fontsize=12)
        axs[0, 0].set_ylim([0, 100])
        axs[0, 1].set_ylabel("Scores %", fontsize=12)
        axs[0, 1].set_title("Recall", fontsize=12)
        axs[0, 1].set_ylim([0, 100])
        axs[1, 0].set_ylabel("Scores %", fontsize=12)
        axs[1, 0].set_title("F1-Score", fontsize=12)
        axs[1, 0].set_ylim([0, 100])
        axs[1, 1].set_ylabel("Time (sec)", fontsize=12)
        axs[1, 1].set_title("Execution time", fontsize=12)
    for w in workflows:
        workflow_df.loc[len(workflow_df)] = [w.name, w.f1[-1], w.recall[-1], w.precision[-1], w.workflow_exec_time]

    if with_visualization:
        axs[0, 0].bar(workflow_df['Name'], workflow_df['Precision'], label=workflow_df['Name'], color='b')
        axs[0, 1].bar(workflow_df['Name'], workflow_df['Recall'], label=workflow_df['Name'], color='g')
        axs[1, 0].bar(workflow_df['Name'], workflow_df['F1'], color='orange')
        axs[1, 1].bar(workflow_df['Name'], workflow_df['Runtime (sec)'], color='r')
    fig.autofmt_xdate()
    plt.show()
    return workflow_df

class OptimizeWorkflow:
    """Optuna Framework for GridSearch/RandomSearch/Prunning in a given pyjedai workflow.
    """

    _id = count()

    def __init__(
            self,
            block_building: Callable, # Mandatory: one method
            entity_matching: Callable, # Mandatory: one method
            block_cleaning: List[Callable] = None, # Optional: multiple methods
            comparison_cleaning: List[Callable] = None, # Optional: multiple methods
            clustering: Callable = None, # Optional: One method
            joins: Callable = None, # Optional: One method
            name: str = None
    ) -> None:
        self.block_cleaning, self.block_building, self.comparison_cleaning, \
            self.clustering, self.joins, self.entity_matching = \
            block_cleaning, block_building, comparison_cleaning, clustering, joins, entity_matching
        self.f1: float
        self.recall: float
        self.precision: float
        self.runtime: float
        self.configurations: float
        self.workflow_exec_time: float
        self._id: int = next(self._id)
        self.name: str = name if name else "OptWorkflow-" + str(self._id)
        self._workflow_bar: tqdm

    def objective(self, target_score: str = 'f1') -> any:
        """Optuna objective function

            Returns:
                 One or more from F1,Recall,Precision
        """
        pass
        # if target_score  == 'f1':
        #     return f1
        # elif target_score  == 'recall':
        #     return recall
        # else:
        #     return precision

    def run(
            self,
            data: Data,
            num_of_trials = 30,
            pruner: optuna.pruners = optuna.pruners.NopPruner,
            sampler: optuna.samplers = optuna.samplers.BaseSampler,
            study_name = "pyjedai_study",
            storage_name = "pyjedai_storage_trials",
            target_score = "f1",
            load_if_exists=True,
            verbose=False,
            tqdm_disable=False,
            workflow_tqdm_enable=False,
            optuna_tqdm_enable=True
    ) -> pd.DataFrame:
        """Executes the experiments based on a workflow

        Args:
            data (Data): _description_
            num_of_trials (int, optional): _description_. Defaults to 30.
            pruner (optuna.pruners, optional): _description_. Defaults to optuna.pruners.NopPruner.
            sampler (optuna.samplers, optional): _description_. Defaults to optuna.samplers.BaseSampler.
            study_name (str, optional): _description_. Defaults to "pyjedai_study".
            storage_name (str, optional): _description_. Defaults to "pyjedai_storage_trials".
            target_score (str, optional): _description_. Defaults to "f1".
            load_if_exists (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to False.
            tqdm_disable (bool, optional): _description_. Defaults to False.
            workflow_tqdm_enable (bool, optional): _description_. Defaults to False.
            optuna_tqdm_enable (bool, optional): _description_. Defaults to True.

        Returns:
            pd.DataFrame: _description_
        """
        study = optuna.create_study(
            directions=["maximize"],
            study_name=study_name,
            storage=storage_name,
            load_if_exists=load_if_exists
        )
        print("Optuna trials starting")
        study.optimize(
            self.objective, 
            n_trials=num_of_trials, 
            show_progress_bar=optuna_tqdm_enable
        )
        print("Optuna trials finished")

    def visualize(self, with_plotly=True):
        pass
    
    def get_best_trial(self):
        pass
    
    def to_df():
        pass
    
    
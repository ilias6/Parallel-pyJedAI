"""Evaluation module
This file contains all the methods for evaluating every module in pyjedai.
"""
from decimal import DivisionByZero
from typing import Type
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from .datamodel import Data
from .utils import are_matching


class Evaluation:
    """Evaluation class. Contains multiple methods for all the fitted & predicted data.
    """
    def __init__(self, data) -> None:
        self.f1: float
        self.recall: float
        self.precision: float
        self.num_of_comparisons: int
        self.total_matching_pairs: int
        self.true_positives: int
        self.true_negatives: int
        self.false_positives: int
        self.false_negatives: int
        self.total_matching_pairs = 0
        self.num_of_true_duplicates: int
        self.data: Data = data

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " + 
                    "Data object has not been initialized with the ground-truth file")

        self.true_positives = self.true_negatives = self.false_positives = self.false_negatives = 0
        self.all_gt_ids = set(self.data._ids_mapping_1.values()) if self.data.is_dirty_er else \
                        set(self.data._ids_mapping_1.values()).union(set(self.data._ids_mapping_2.values()))

    def _set_true_positives(self, true_positives) -> None:
        self.true_positives = true_positives

    def _set_total_matching_pairs(self, total_matching_pairs) -> None:
        self.total_matching_pairs = total_matching_pairs

    
    # if isinstance(prediction, dict) and isinstance(list(prediction.values())[0], set):
    #     # case of candidate pairs, entity-id -> {entity-id, ..}
    #     self.total_matching_pairs = sum([len(block) for block in prediction.values()])
    #     for _, (id1, id2) in gt.iterrows():
    #         id1 = self.data._ids_mapping_1[id1]
    #         id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er else self.data._ids_mapping_2[id2]
    #         if (id1 in prediction and id2 in prediction[id1]) or   \
    #             (id2 in prediction and id1 in prediction[id2]):
    #             self.true_positives += 1
    # elif isinstance(prediction, nx.Graph):
    #     self.total_matching_pairs = prediction.number_of_edges()
    #     for _, (id1, id2) in gt.iterrows():
    #         id1 = self.data._ids_mapping_1[id1]
    #         id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er else self.data._ids_mapping_2[id2]
    #         if (id1 in prediction and id2 in prediction[id1]) or   \
    #              (id2 in prediction and id1 in prediction[id2]):
    #             self.true_positives += 1
    # else: # blocks, clusters evaluation
    #     entity_index: dict = self._create_entity_index(prediction, all_gt_ids)
    #     for _, (id1, id2) in gt.iterrows():
    #         id1 = self.data._ids_mapping_1[id1]
    #         id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er else self.data._ids_mapping_2[id2]
    #         if id1 in entity_index and    \
    #             id2 in entity_index and     \
    #                 are_matching(entity_index, id1, id2):
    #             self.true_positives += 1

    def calculate_scores(self, true_positives=None) -> None:
        if true_positives is not None:
            self.true_positives = true_positives

        if self.total_matching_pairs == 0:
            warn("Evaluation: No matches found", Warning)
            self.num_of_true_duplicates = self.false_negatives \
                = self.false_positives = self.total_matching_pairs \
                    = self.true_positives = self.true_negatives \
                        = self.recall = self.f1 = self.precision = 0
        else:
            self.num_of_true_duplicates = len(self.data.ground_truth)
            self.false_negatives = self.num_of_true_duplicates - self.true_positives
            self.false_positives = self.total_matching_pairs - self.true_positives
            cardinality = (self.data.num_of_entities_1*(self.data.num_of_entities_1-1))/2 \
                if self.data.is_dirty_er else self.data.num_of_entities_1 * self.data.num_of_entities_2
            self.true_negatives = cardinality - self.false_negatives - self.false_positives
            self.precision = self.true_positives / self.total_matching_pairs
            self.recall = self.true_positives / self.num_of_true_duplicates
            if self.precision == 0.0 or self.recall == 0.0:
                raise DivisionByZero("Recall or Precision is equal to zero. Can't calculate F1 score.")
            else:
                self.f1 = 2*((self.precision*self.recall)/(self.precision+self.recall))

    def report(
            self,
            configuration: dict = None,
            export_to_df=False,
            export_to_dict=False,
            with_classification_report=False,
            verbose=True
        ) -> any:

        results_dict = {
                'Precision %': self.precision*100,
                'Recall %': self.recall*100,
                'F1 %': self.f1*100,
                'True Positives': self.true_positives,
                'False Positives': self.false_positives,
                'True Negatives': self.true_negatives,
                'False Negatives': self.false_negatives
            }

        if verbose:
            if configuration:
                print('*' * 123)
                print(' ' * 40, 'Μethod: ', configuration['name'])
                print('*' * 123)
                print(
                    "Method name: " + configuration['name'] +
                    "\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in configuration['parameters'].items()]) +
                    "Runtime: {:2.4f} seconds".format(configuration['runtime'])
                )
            else:
                print(" " + (configuration['name'] if configuration else "") + " Evaluation \n---")


            print(u'\u2500' * 123)
            print("Performance:\n\tPrecision: {:9.2f}% \n\tRecall:    {:9.2f}%\n\tF1-score:  {:9.2f}%".format(self.precision*100, self.recall*100, self.f1*100))
            print(u'\u2500' * 123)
            if with_classification_report:
                print("Classification report:\n\tTrue positives: {:d}\n\tFalse positives: {:d}\n\tTrue negatives: {:d}\n\tFalse negatives: {:d}\n\tTotal comparisons: {:d}".format(
                    int(self.true_positives), int(self.false_positives), int(self.true_negatives), \
                    int(self.false_negatives), int(self.total_matching_pairs))
                )
                print(u'\u2500' * 123)

        if export_to_df:
            pd.set_option("display.precision", 2)
            results = pd.DataFrame.from_dict(results_dict, orient='index').T
            return results

        return results_dict

    def _create_entity_index_from_clusters(
            self,
            clusters: list,
            all_ground_truth_ids: set
    ) -> dict:
        entity_index = dict()
        for cluster, cluster_id in zip(clusters, range(0, len(clusters))):
            cluster_entities_d1 = 0
            cluster_entities_d2 = 0
            for entity_id in cluster.intersection(all_ground_truth_ids):
                entity_index[entity_id] = cluster_id

                if not self.data.is_dirty_er:
                    if entity_id < self.data.dataset_limit:
                        cluster_entities_d1 += 1
                    else:
                        cluster_entities_d2 += 1

            if self.data.is_dirty_er:
                self.total_matching_pairs += len(cluster)*(len(cluster)-1)/2
            else:
                self.total_matching_pairs += cluster_entities_d1*cluster_entities_d2

        return entity_index

    def _create_entity_index_from_blocks(
            self,
            blocks: dict
    ) -> dict:
        entity_index = dict()
        for block_id, block in blocks.items():       
            for entity_id in block.entities_D1:
                entity_index.setdefault(entity_id, set())
                entity_index[entity_id].add(block_id)

            if not self.data.is_dirty_er:
                for entity_id in block.entities_D2:
                    entity_index.setdefault(entity_id, set())
                    entity_index[entity_id].add(block_id)

            if self.data.is_dirty_er:
                self.total_matching_pairs += len(block.entities_D1)*(len(block.entities_D1)-1)/2
            else:
                self.total_matching_pairs += len(block.entities_D1)*len(block.entities_D2)

        return entity_index

    def confusion_matrix(self):
        """Generates a confusion matrix based on the classification report.
        """
        heatmap = [
            [int(self.true_positives), int(self.false_positives)],
            [int(self.false_negatives), int(self.true_negatives)]
        ]
        # plt.colorbar(heatmap)
        sns.heatmap(
            heatmap,
            annot=True,
            cmap='Blues',
            xticklabels=['Non-Matching', 'Matching'],
            yticklabels=['Non-Matching', 'Matching'],
            fmt='g'
        )
        plt.title("Confusion Matrix", fontsize=12, fontweight='bold')
        plt.xlabel("Predicted pairs", fontsize=10, fontweight='bold')
        plt.ylabel("Real matching pairs", fontsize=10, fontweight='bold')
        plt.show()

def write(
        prediction: any,
        data: Data
    ) -> pd.DataFrame:
    """creates a dataframe for the evaluation report

    Args:
        prediction (any): Predicted pairs, blocks, candidate pairs or graph
        data (Data): initial dataset

    Returns:
        pd.DataFrame: Dataframe containg evaluation scores and stats
    """
    if data.ground_truth is None:
        raise AttributeError("Can not proceed to evaluation without a ground-truth file. \
            Data object mush have initialized with the ground-truth file")
    pairs_df = pd.DataFrame(columns=['id1', 'id2'])
    if isinstance(prediction, list): # clusters evaluation
        for cluster in prediction:
            lcluster = list(cluster)
            for i1 in range(0, len(lcluster)):
                for i2 in range(i1+1, len(lcluster)):
                    id1 = data._gt_to_ids_reversed_1[lcluster[i1]]
                    id2 = data._gt_to_ids_reversed_1[lcluster[i2]] if data.is_dirty_er \
                            else data._gt_to_ids_reversed_2[lcluster[i2]]
                    pairs_df = pd.concat(
                        [pairs_df, pd.DataFrame([{'id1':id1, 'id2':id2}], index=[0])], 
                        ignore_index=True
                    )
    elif 'Block' in str(type(list(prediction.values())[0])): # blocks evaluation
        for _, block in prediction.items():
            if data.is_dirty_er:
                lblock = list(block.entities_D1)
                for i1 in range(0, len(lblock)):
                    for i2 in range(i1+1, len(lblock)):
                        id1 = data._gt_to_ids_reversed_1[lblock[i1]]
                        id2 = data._gt_to_ids_reversed_1[lblock[i2]] if data.is_dirty_er \
                            else data._gt_to_ids_reversed_2[lblock[i2]]
                        pairs_df = pd.concat([pairs_df, pd.DataFrame([{'id1':id1, 'id2':id2}], index=[0])], ignore_index=True)
            else:
                for i1 in block.entities_D1:
                    for i2 in block.entities_D2:
                        id1 = data._gt_to_ids_reversed_1[i1]
                        id2 = data._gt_to_ids_reversed_1[i2] if data.is_dirty_er \
                            else data._gt_to_ids_reversed_2[i2]
                        pairs_df = pd.concat([pairs_df, pd.DataFrame([{'id1':id1, 'id2':id2}], index=[0])], ignore_index=True)
    elif isinstance(prediction, dict) and isinstance(list(prediction.values())[0], set):# candidate pairs
        for entity_id, candidates in prediction:
            id1 = data._gt_to_ids_reversed_1[entity_id]                                            
            for candiadate_id in candidates:
                id2 = data._gt_to_ids_reversed_1[candiadate_id] if data.is_dirty_er \
                        else data._gt_to_ids_reversed_2[candiadate_id]
                pairs_df = pd.concat([pairs_df, pd.DataFrame([{'id1':id1, 'id2':id2}], index=[0])], ignore_index=True)
    elif isinstance(prediction, nx.Graph): # graph
        for edge in prediction.edges:
            id1 = data._gt_to_ids_reversed_1[edge[0]]
            id2 = data._gt_to_ids_reversed_1[edge[1]] if data.is_dirty_er \
                        else data._gt_to_ids_reversed_2[edge[1]]
            pairs_df = pd.concat([pairs_df, pd.DataFrame([{'id1':id1, 'id2':id2}], index=[0])], ignore_index=True)
    else:
        raise TypeError("Not supported type")

    return pairs_df

import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pyjedai.block_building import (ExtendedQGramsBlocking,
                                    ExtendedSuffixArraysBlocking,
                                    QGramsBlocking, StandardBlocking,
                                    SuffixArraysBlocking)
from pyjedai.block_cleaning import BlockFiltering, BlockPurging
from pyjedai.clustering import ConnectedComponentsClustering
from pyjedai.comparison_cleaning import (BLAST, CardinalityEdgePruning,
                                         CardinalityNodePruning,
                                         ComparisonPropagation,
                                         ReciprocalCardinalityNodePruning,
                                         ReciprocalWeightedNodePruning,
                                         WeightedEdgePruning,
                                         WeightedNodePruning)
from pyjedai.datamodel import Data
from pyjedai.matching import EntityMatching
from pyjedai.workflow import WorkFlow, compare_workflows
from pyjedai.evaluation import Evaluation

directory = r'../data/der/synthetic'
benchmarks_df = pd.DataFrame(columns=['Algorithm', 'Recall', 'Precision', 'F1', 'Runtime'])
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if "D" in f:
        # print(f)
        X_filepath = f+"/X.csv"
        GT_filepath = f+"/GT.csv"
        print("\n------\nReading dataset: "+f)
        # print(X_filepath)
        # print(GT_filepath)
        data = Data(
            dataset_1=pd.read_csv(
                X_filepath,
                sep='|', 
                engine='python', 
                na_filter=False).astype(str),
            id_column_name_1='Id',
            ground_truth=pd.read_csv(GT_filepath,
                                     sep='|', 
                                     engine='python'),
        )
        results = []
        e = Evaluation(data)

        sb = StandardBlocking()
        blocks, entity_index = sb.build_blocks(data)
        res = e.report(blocks, to_df=True)
        results.append(['StandardBlocking', res['F1'], res['Recall'], res['Precision'], sb.execution_time])

        bp = BlockPurging(smoothing_factor=1.0)
        bp_blocks = bp.process(blocks, data)
        res = e.report(bp_blocks, to_df=True)
        results.append(['BlockPurging', res['F1'], res['Recall'], res['Precision'], bp.execution_time])

        bf = BlockFiltering()
        bf_blocks = bf.process(blocks, data)
        res = e.report(bp_blocks, to_df=True)
        results.append(['BlockFiltering', res['F1'], res['Recall'], res['Precision'], bf.execution_time])


        cnp = CardinalityNodePruning(weighting_scheme='JS')
        candidate_pairs_blocks = cnp.process(bf_blocks, data)
        res = e.report(bp_blocks, to_df=True)
        results.append(['CardinalityNodePruning', res['F1'], res['Recall'], res['Precision'], cnp.execution_time])

        em = EntityMatching(metric='cosine', tokenizer='qgram_tokenizer', qgram=4)
        em_graph = em.predict(candidate_pairs_blocks, data)
        res = e.report(em_graph, to_df=True)
        results.append(['EntityMatching', res['F1'], res['Recall'], res['Precision'], em.execution_time])

        ccc = ConnectedComponentsClustering()
        clusters = ccc.process(em_graph)
        res = e.report(clusters, to_df=True)
        results.append(['ConnectedComponentsClustering', res['F1'], res['Recall'], res['Precision'], ccc.execution_time])
        
        benchmarks_df = pd.DataFrame(results, columns=['Algorithm', 'F1', 'Recall', 'Precision', 'Time'])
        benchmarks_df.to_csv('release-04-05.csv')
        
        
        
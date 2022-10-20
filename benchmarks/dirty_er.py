import os
import sys
import re
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pyjedai.block_building import StandardBlocking
from pyjedai.block_cleaning import BlockFiltering, BlockPurging
from pyjedai.clustering import ConnectedComponentsClustering
from pyjedai.comparison_cleaning import CardinalityNodePruning
from pyjedai.datamodel import Data
from pyjedai.matching import EntityMatching
from pyjedai.evaluation import Evaluation

directory = r'../data/der/synthetic'
try:
    results = []
    i=0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if "D" in f:
            X_filepath = f+"/X.csv"
            GT_filepath = f+"/GT.csv"
            dataset_name =  f[f.find('D'):]
            print("\n------\nReading dataset: "+f)
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
            e = Evaluation(data)
            sb = StandardBlocking()
            blocks, entity_index = sb.build_blocks(data)
            total_time = 0.0
            res = e.report(blocks, to_df=True)
            results.append([dataset_name, 'StandardBlocking', res['F1 %'], res['Recall %'], res['Precision %'], sb.execution_time])
            total_time += sb.execution_time
            
            bp = BlockPurging(smoothing_factor=1.0)
            bp_blocks = bp.process(blocks, data)
            res = e.report(bp_blocks, to_df=True)
            results.append([dataset_name, 'BlockPurging', res['F1 %'], res['Recall %'], res['Precision %'], bp.execution_time])
            total_time += sb.execution_time

            bf = BlockFiltering()
            bf_blocks, entity_index = bf.process(blocks, data)
            res = e.report(bp_blocks, to_df=True)
            results.append([dataset_name, 'BlockFiltering', res['F1 %'], res['Recall %'], res['Precision %'], bf.execution_time])
            total_time += sb.execution_time

            cnp = CardinalityNodePruning(weighting_scheme='JS')
            candidate_pairs_blocks = cnp.process(bf_blocks, data, entity_index)
            res = e.report(bp_blocks, to_df=True)
            results.append([dataset_name, 'CardinalityNodePruning', res['F1 %'], res['Recall %'], res['Precision %'], cnp.execution_time])
            total_time += sb.execution_time

            em = EntityMatching(metric='cosine', tokenizer='qgram_tokenizer', qgram=4)
            em_graph = em.predict(candidate_pairs_blocks, data)
            res = e.report(em_graph, to_df=True)
            results.append([dataset_name, 'EntityMatching', res['F1 %'], res['Recall %'], res['Precision %'], em.execution_time])
            total_time += sb.execution_time

            ccc = ConnectedComponentsClustering()
            clusters = ccc.process(em_graph)
            res = e.report(clusters, to_df=True)
            results.append([dataset_name, 'ConnectedComponentsClustering', res['F1 %'], res['Recall %'], res['Precision %'], ccc.execution_time])
            total_time += sb.execution_time

            results.append([dataset_name, 'Total', res['F1 %'], res['Recall %'], res['Precision %'], total_time])
            
            i+=1
            if i == 2:
                break

    benchmarks_df = pd.DataFrame(results, columns=['dataset', 'workflow_step', 'f1', 'recall', 'precision', 'time'])
    benchmarks_df.to_csv('release_04to05.csv')
except KeyboardInterrupt:
    benchmarks_df = pd.DataFrame(results, columns=['dataset', 'workflow_step', 'f1', 'recall', 'precision', 'time'])
    benchmarks_df.to_csv('release_04to05_stoped.csv')
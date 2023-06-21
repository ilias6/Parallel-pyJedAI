import time
import os
import sys
import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.workflow import WorkFlow, compare_workflows
from pyjedai.block_building import StandardBlocking, QGramsBlocking, ExtendedQGramsBlocking, SuffixArraysBlocking, ExtendedSuffixArraysBlocking
from pyjedai.block_cleaning import BlockFiltering, BlockPurging
from pyjedai.comparison_cleaning import WeightedEdgePruning, WeightedNodePruning, CardinalityEdgePruning, CardinalityNodePruning, BLAST, ReciprocalCardinalityNodePruning, ReciprocalWeightedNodePruning, ComparisonPropagation
from pyjedai.matching import EntityMatching
from pyjedai.clustering import ConnectedComponentsClustering, UniqueMappingClustering
import numpy as np

D1CSV = [
    "rest1.csv", "abt.csv", "amazon.csv", "dblp.csv",  "imdb.csv",  "imdb.csv",  "tmdb.csv",  "walmart.csv",   "dblp.csv",    "imdb.csv"
]
D2CSV = [
    "rest2.csv", "buy.csv", "gp.csv",     "acm.csv",   "tmdb.csv",  "tvdb.csv",  "tvdb.csv",  "amazon.csv",  "scholar.csv", "dbpedia.csv"
]
GTCSV = [
    "gt.csv",   "gt.csv",   "gt.csv",     "gt.csv",   "gt.csv", "gt.csv", "gt.csv", "gt.csv", "gt.csv", "gt.csv"
]
D = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9','D10']

separator = [
    '|', '|', '#', '\\%', '|', '|', '|', '|', '>', '|'
]
engine = [
    'python', 'python','python','python','python','python','python','python','python', None
]

set_metrics = [
    'cosine', 'dice', 'jaccard', 'overlap_coefficient'
]

bag_metrics = [
    'tf-idf'
]

tokenizers = [
    'white_space_tokenizer', 'char_qgram_tokenizer', 'word_qgram_tokenizer'
]

available_metrics = set_metrics + bag_metrics

weighting_schemes = {1:'JS',2:'CBS',7:'ARCS'}


datasets_wanted = [1, 2, 7]
for i in datasets_wanted:
    print("\n\nDataset: ", D[i])
    trial = 0
    d = D[i]
    d1 = D1CSV[i]
    d2 = D2CSV[i]
    gt = GTCSV[i]
    s = separator[i]
    e = engine[i]

    # Create a csv file 
    with open(d+'_bw.csv', 'w') as f:
        f.write('trial, metric, tokenizer, qgram, threshold, precision, recall, f1, runtime\n')
        data = Data(dataset_1=pd.read_csv("./data/ccer/" + d + "/" + d1 , sep=s, engine=e, na_filter=False).astype(str),
                    id_column_name_1='id',
                    dataset_2=pd.read_csv("./data/ccer/" + d + "/" + d2 , sep=s, engine=e, na_filter=False).astype(str),
                    id_column_name_2='id',
                    ground_truth=pd.read_csv("./data/ccer/" + d + "/gt.csv", sep=s, engine=e))

        if 'aggregated value' in data.attributes_1:
            data.dataset_1 = data.dataset_1.drop(columns=['aggregated value'], inplace=True)
        
        if 'aggregated value' in data.attributes_2:
            data.dataset_2 = data.dataset_2.drop(columns=['aggregated value'], inplace=True)

        for em_method in available_metrics:
            for tokenizer in tokenizers:
                for q in range(1,6):
                    for thr in np.arange(0, 1, 0.1):
                        try:
                            t1 = time.time()
                            sb = StandardBlocking()
                            blocks = sb.build_blocks(data, tqdm_disable=False)

                            cbbp = BlockPurging(smoothing_factor=1.0)
                            blocks = cbbp.process(blocks, data, tqdm_disable=False)

                            bf = BlockFiltering(ratio=0.8)
                            blocks = bf.process(blocks, data, tqdm_disable=False)

                            wep = CardinalityNodePruning(weighting_scheme=weighting_schemes[i])
                            candidate_pairs_blocks = wep.process(blocks, data, tqdm_disable=False)

                            em = EntityMatching(
                                metric=em_method,
                                tokenizer=tokenizer,
                                qgram=q,
                                similarity_threshold=0.0
                            )
                            pairs_graph = em.predict(candidate_pairs_blocks, data, tqdm_disable=False)

                            ccc = UniqueMappingClustering()
                            clusters = ccc.process(pairs_graph, data,similarity_threshold=thr)
                            results = ccc.evaluate(clusters, with_classification_report=True, verbose=True)

                            t2 = time.time()
                            f1, precision, recall = results['F1 %'], results['Precision %'], results['Recall %']

                            f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(trial, em.metric, tokenizer, q, ccc.similarity_threshold, precision, recall, f1, t2-t1))
                        
                        except ValueError as e:

                            # Handle the exception and force Optuna to continue
                            f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(trial, str(e), None, None, None, None, None, None, None))                    
                        
                        trial += 1
    f.close()

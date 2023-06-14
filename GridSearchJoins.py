import time
import os
import sys
import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.joins import EJoin
from pyjedai.clustering import UniqueMappingClustering
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

schema_based_attributes = {1:["name"], 2:["title"], 7:["modelno"]}
thresholds = [0.40, 0.45, 0.90]
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
    with open(d+'_joins.csv', 'w') as f:
        f.write('trial, metric, tokenization, qgram, threshold, precision, recall, f1, runtime\n')
        data = Data(
            dataset_1=pd.read_csv("./data/ccer/" + d + "/" + d1 , 
                                sep=s,
                                engine=e,
                                na_filter=False).astype(str),
            id_column_name_1='id',
            dataset_2=pd.read_csv("./data/ccer/" + d + "/" + d2 , 
                                sep=s, 
                                engine=e, 
                                na_filter=False).astype(str),
            id_column_name_2='id',
            ground_truth=pd.read_csv("./data/ccer/" + d + "/gt.csv", sep=s, engine=e))

        if 'aggregated value' in data.attributes_1:
            data.dataset_1 = data.dataset_1.drop(columns=['aggregated value'], inplace=True)
        
        if 'aggregated value' in data.attributes_2:
            data.dataset_2 = data.dataset_2.drop(columns=['aggregated value'], inplace=True)


        tokenizations = ['qgrams','standard','standard_multiset','qgrams_multiset']
        metrics = ['dice', 'cosine', 'jaccard']
        

        for m in metrics:
            for t in tokenizations:                        
                for q in range(1, 5):
                    for thr in np.arange(0, 1, 0.1):
                        trial += 1
                        try:
                            print("\n\nTrial: ", trial, "Metric: ", m, "Tokenization: ", t, "Qgram: ", q, "Threshold: ", thr)
                            t1 = time.time()
                            ejoin = EJoin(metric = m, tokenization = t, qgrams = q, similarity_threshold=0.0)
                            g = ejoin.fit(data, attributes_1=schema_based_attributes[i], attributes_2=schema_based_attributes[i])
                            ejoin.evaluate(g)

                            ccc = UniqueMappingClustering()
                            clusters = ccc.process(g, data,similarity_threshold=thr)
                            results = ccc.evaluate(clusters, with_classification_report=True, verbose=True)

                            t2 = time.time()
                            f1, precision, recall = results['F1 %'], results['Precision %'], results['Recall %']

                            f.write('{}, {}, {}, {}, {}, {},{}\n'.format(trial, m, t, q, thr, precision, recall, f1, t2-t1))
                        
                        except ValueError as e:

                            # Handle the exception and force Optuna to continue
                            f.write('{}, {}, {}, {}, {}, {},{}\n'.format(trial, str(e), None, None, None, None, None))                    
    f.close()

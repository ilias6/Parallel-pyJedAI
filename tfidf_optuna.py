import time
import optuna
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
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

db_name = "pyjedai_tfidf"
storage_name = "sqlite:///{}.db".format(db_name)

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


for i in range(0,len(D1CSV)):
    print("\n\nDataset: ", D[i])

    d = D[i]
    d1 = D1CSV[i]
    d2 = D2CSV[i]
    gt = GTCSV[i]
    s = separator[i]
    e = engine[i]

    # Create a csv file 
    with open(d+'_tfidf_optuna_em.csv', 'w') as f:
        f.write('trial, metric, threshold, tokenizer, metric, qgram, precision, recall, f1, em_f1, runtime\n')
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

        title = d + "_tfidf_entity_matching"
        study_name = title  # Unique identifier of the study.

        '''
        OPTUNA objective function
        '''
        def objective(trial):
            try:
                t1 = time.time()
                sb = StandardBlocking()
                blocks = sb.build_blocks(data, tqdm_disable=False)

                cbbp = BlockPurging(smoothing_factor=1.0)
                blocks = cbbp.process(blocks, data, tqdm_disable=False)

                bf = BlockFiltering(ratio=0.8)
                blocks = bf.process(blocks, data, tqdm_disable=False)

                wep = CardinalityNodePruning(weighting_scheme='JS')
                candidate_pairs_blocks = wep.process(blocks, data, tqdm_disable=False)
                wep.evaluate(candidate_pairs_blocks)
                
                em = EntityMatching(metric='tf-idf', 
                                    tokenizer = trial.suggest_categorical("tokenizer", ["char_qgram_tokenizer", "word_qgram_tokenizer"]), 
                                    qgram=trial.suggest_int("qgram", 1, 5),
                                    tfidf_similarity_metric=trial.suggest_categorical("tfidf_similarity_metric", ["cosine", "jaccard", "dice"]), 
                                    similarity_threshold=0.0
                )
                pairs_graph = em.predict(candidate_pairs_blocks, data, tqdm_disable=False)
                em_results = em.evaluate(pairs_graph, data)
                
                thresholds = [em.get_weights_avg(), em.get_weights_median(), em.get_weights_avg()+em.get_weights_standard_deviation(), em.get_weights_median()+em.get_weights_standard_deviation()]
                
                ccc = UniqueMappingClustering()
                clusters = ccc.process(pairs_graph, data, similarity_threshold=trial.suggest_categorical("similarity_threshold", thresholds))

                results = ccc.evaluate(clusters, with_classification_report=True, verbose=True)

                t2 = time.time()
                f1, precision, recall = results['F1 %'], results['Precision %'], results['Recall %']

                f.write('{}, {}, {}, {}, {}, {},{}\n'.format(trial.number, em.metric, ccc.similarity_threshold, em.tokenizer, em.tfidf_similarity_metric, em.qgram, precision, recall, f1, em_results['F1 %'], t2-t1))
            
                return f1

            except ValueError as e:
                # Handle the exception and force Optuna to continue
                
                print(e)
                trial.set_user_attr("failed", True)
                f.write('{}, {}, {}, {}, {}, {},{}\n'.format(trial.number, str(e), None, None,None,None, None, None, None, None, None))
                return optuna.TrialPruned()
        
        study_name = title  # Unique identifier of the study.
        num_of_trials = 1
        study = optuna.create_study(
            directions=["maximize"],
            study_name=study_name,
            storage=storage_name,
            load_if_exists=False
        )
        print("Optuna trials starting")
        study.optimize(
            objective, 
            n_trials=num_of_trials, 
            show_progress_bar=True,
            callbacks=[MaxTrialsCallback(num_of_trials, states=(TrialState.COMPLETE,))]
        )
        print("Optuna trials finished")

    f.close()
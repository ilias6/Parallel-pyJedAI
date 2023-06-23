import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.joins import EJoin
from pyjedai.clustering import UniqueMappingClustering
import time

D1CSV = [ "abt.csv", "amazon.csv", "walmart.csv"]
D2CSV = ["buy.csv", "gp.csv", "amazon.csv"]
GTCSV = [ "gt.csv", "gt.csv", "gt.csv"]
D = ['D2', 'D3', 'D8']

separator = ['|', '#', '|']

schema_based_attributes = [["name"], ["title"], ["modelno"]]
thresholds = [0.40, 0.45, 0.90]
metric = ['cosine', 'cosine', 'jaccard']
tokenizations = ['qgrams', 'qgrams', 'standard']
gram_sizes = [2, 3 , 1]
thresholds = [0.4 , 0.2 , 0.6]

for i in range(0, len(D)):
    print("\n\nDataset: ", D[i])
    d = D[i]
    data = Data(dataset_1=pd.read_csv(d + D1CSV[i] , sep=separator[i], engine='python', na_filter=False).astype(str),
                id_column_name_1='id',
                dataset_2=pd.read_csv(d + D2CSV[i] , sep=separator[i], engine='python', na_filter=False).astype(str),
                id_column_name_2='id',
                ground_truth=pd.read_csv(d + "gt.csv", sep=separator[i], engine='python'))

    averageRt = 0
    for iteration in range(0, 10):
        time1 = time.time()
    
        ejoin = EJoin(metric = metric[i], 
                      tokenization = tokenizations[i], 
                      qgrams = gram_sizes[i], 
                      similarity_threshold=0.0)
        g = ejoin.fit(data, 
                      attributes_1=schema_based_attributes[i], 
                      attributes_2=schema_based_attributes[i])
        ejoin.evaluate(g)
    
        ccc = UniqueMappingClustering()
        clusters = ccc.process(g, data,similarity_threshold=thresholds[i])
        clp = ccc.evaluate(clusters, with_classification_report=True, verbose=True)
        clp.setStatistics();
        
        time2 = time.time()
        averageRt += (time2 - time1)
        print("\nScores:\n\tRecall: ", clp.recall,
              "\n\tPrecision: ",  clp.precision,
              "\n\tF-Measure: ",  clp.fMeasure,
              "\n\tRuntime: ", time2 - time1)
    
    print("Average RT\t:\t" + str(averageRt/10))
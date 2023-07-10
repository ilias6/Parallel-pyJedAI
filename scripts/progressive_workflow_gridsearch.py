import time
import os
import sys
import pandas as pd
import numpy as np
import json
from itertools import product

from pyjedai.datamodel import Data
from pyjedai.workflow import ProgressiveWorkFlow
from pyjedai.utils import get_class, values_given, get_multiples, necessary_dfs_supplied, store_workflow_results
from pyjedai.block_building import (StandardBlocking,
                                    QGramsBlocking,
                                    ExtendedQGramsBlocking,
                                    SuffixArraysBlocking,
                                    ExtendedSuffixArraysBlocking)
                                   
from pyjedai.block_cleaning import (BlockFiltering,
                                    BlockPurging)                         
from pyjedai.comparison_cleaning import (WeightedEdgePruning, 
                                         WeightedNodePruning,
                                         CardinalityEdgePruning,
                                         CardinalityNodePruning, 
                                         BLAST,
                                         ReciprocalCardinalityNodePruning,
                                         ReciprocalWeightedNodePruning,
                                         ComparisonPropagation)                                   
from pyjedai.prioritization import (GlobalTopPM, 
                                    LocalTopPM, 
                                    EmbeddingsNNBPM, 
                                    GlobalPSNM, 
                                    LocalPSNM, 
                                    PESM, 
                                    WhooshPM)
from pyjedai.evaluation import Evaluation

#-EDIT-THOSE-#

# parameters native to the pyjedai progressive workflow
# don't edit, unless new parameters were added to the workflow
valid_workflow_parameters = ['matcher',
                            'algorithm',
                            'number_of_nearest_neighbors',
                            'indexing',
                            'similarity_function',
                            'language_model',
                            'tokenizer',
                            'weighting_scheme',
                            'window_size']
# path of the configuration file
CONFIG_FILE_PATH = '~/home/jm/pyJedAI/pyJedAI-Dev/script-configs/per_experiments.json'
# which configuration from the json file should be used in current experiment  
EXPERIMENT_NAME = 'vector-based-debug-test'
# path at which the results will be stored within a json file
RESULTS_STORE_PATH = '~/home/jm/pyJedAI/pyJedAI-Dev/scirpts-results/' + EXPERIMENT_NAME + '.json'
# results should be stored in the predefined path
STORE_RESULTS = True
# AUC calculation and ROC visualization after execution
VISUALIZE_RESULTS = True
# workflow arguments and execution info should be printed in terminal once executed
PRINT_WORKFLOWS = True
# identifier column names for source and target datasets
D1_ID = 'id'
D2_ID = 'id'

##############                          
                                   
with open(CONFIG_FILE_PATH) as file:
    config = json.load(file)
    
config = config[EXPERIMENT_NAME]
workflow_config = {k: v for k, v in config.items() if(values_given(v) and v in valid_workflow_parameters)}
workflow_parameters = list(workflow_config.keys())
workflow_values = list(workflow_config.values())
workflow_combinations = list(product(*workflow_values))

if(not necessary_dfs_supplied(config)):
    raise ValueError("Different number of source, target dataset and ground truth paths!")

datasets_info = list(zip(config['source_dataset_path'], config['target_dataset_path'], config['ground_truth_path']))


results = dict()
execution_count : int = 0

for id, dataset_info in enumerate(datasets_info):
    dataset_id = id + 1
    d1_path, d2_path, gt_path = dataset_info
    dataset_name = config['dataset_name'][id] if(values_given(config, 'dataset_name') and len(config['dataset_name']) > id) else ("D" + str(dataset_id))
    
    d1 = pd.read_csv(d1_path, sep='|', engine='python', na_filter=False).astype(str)
    d2 = pd.read_csv(d2_path, sep='|', engine='python', na_filter=False).astype(str)
    gt = pd.read_csv(gt_path, sep='|', engine='python')

    d1_attributes = config['d1_attributes'][id] if values_given(config, 'd1_attributes') else d1.columns.tolist()
    d2_attributes = config['d2_attributes'][id] if values_given(config, 'd2_attributes') else d2.columns.tolist()

    data = Data(
        dataset_1=d1,
        attributes_1=['id','name','description'],
        id_column_name_1=D1_ID,
        dataset_2=d2,
        attributes_2=['id','name','description'],
        id_column_name_2=D2_ID,
        ground_truth=gt,
    )
    
    results = {}
    true_positives_number = len(gt)
    budgets = config['budget'] if values_given(config, 'budget') else get_multiples(true_positives_number, 10)
        
    for budget in budgets:
        for workflow_combination in workflow_combinations:
            workflow_arguments = dict(zip(workflow_parameters, workflow_combination))
            workflow_arguments['budget'] = budget
            workflow_arguments['dataset'] = dataset_name
            
            current_workflow = ProgressiveWorkFlow()
            current_workflow.run(data=data, **workflow_arguments)            
            current_workflow_info : dict =  store_workflow_results(results=results,current_workflow=current_workflow,workflow_arguments=workflow_arguments)
            execution_count += 1
            
            if(PRINT_WORKFLOWS):
                print(f"#### WORKFLOW {execution_count} ####")
                print(current_workflow_info)
            
    
evaluator = Evaluation()
if(VISUALIZE_RESULTS):
    evaluator.visualize_results_roc(results=results)        
                     
if(STORE_RESULTS):
    with open(RESULTS_STORE_PATH, 'w') as file:
        json.dump(results, file, indent=4)  
            
            
    
            
            
            
    
    
    
                                      

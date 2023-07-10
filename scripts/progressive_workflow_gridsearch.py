import time
import os
import sys
import pandas as pd
import numpy as np
import json
from itertools import product

from .datamodel import Data
from .workflow import ProgressiveWorkFlow
from .utils import get_class, values_given, get_multiples, necessary_dfs_supplied
from .block_building import StandardBlocking,
                                   QGramsBlocking,
                                   ExtendedQGramsBlocking,
                                   SuffixArraysBlocking,
                                   ExtendedSuffixArraysBlocking
                                   
from .block_cleaning import BlockFiltering,
                                   BlockPurging
                                   
from .comparison_cleaning import WeightedEdgePruning,
                                        WeightedNodePruning,
                                        CardinalityEdgePruning,
                                        CardinalityNodePruning,
                                        BLAST,
                                        ReciprocalCardinalityNodePruning,
                                        ReciprocalWeightedNodePruning,
                                        ComparisonPropagation
                                        
from .prioritization import GlobalTopPM,
                                   LocalTopPM,
                                   EmbeddingsNNBPM,
                                   GlobalPSNM,
                                   LocalPSNM,
                                   PESM,
                                   WhooshPM

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
true_positives_number = len(gt)
budgets = config['budget'] if values_given(config, 'budget') else get_multiples(true_positives_number, 10)


results = dict()

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
    
    dataset_results = {}
    matcher_results = {}
    
    for budget in budgets:
        for workflow_combination in workflow_combinations:
            workflow_arguments = dict(zip(workflow_parameters, workflow_combination))
            workflow_arguments['budget'] = budget
            workflow_arguments['dataset'] = dataset_name
            
            current_workflow = ProgressiveWorkFlow()
            current_workflow.run(data=data, **workflow_arguments)
            store_workflow_results(results, current_workflow, workflow_arguments)
            
            
            matcher_name = workflow_arguments['matcher']
            dataset_results[dataset_name] = dataset_results[dataset_name] if dataset_name in dataset_results else dict()
            
            matcher_results = dataset_results[dataset_name]
            matcher_results[matcher_name] = matcher_results[matcher_name] if matcher_name in matcher_results else []
            
            matcher_info = matcher_results[matcher_name]
            workflows_info = matcher_info
            if(language_model in workflow_arguments):
                lm_name = workflow_arguments['language_model']
                matcher_info[lm_name] = matcher_info[lm_name] if lm_name in matcher_info else []
                workflows_info = matcher_info[lm_name]
                
            workflow_info = {argument: value for argument, value in workflow_arguments.items() if(argument is not 'matcher' and argument is not 'language_model')} 
                
            
                   
            
            
            info = {argument: value for argument, value in workflow_arguments.items() if(argument is not 'matcher' and argument is not 'language_model')}
            info['candidates'] = current_workflow.progressive_matcher.pairs
            info['recall'] = 
            
            
            
            
            results[dataset_name][workflow_arguments['matcher']][]
            
            
    
            
            
            
    
    
    
                                      

import time
import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
from itertools import product
from pyjedai.utils import to_path
from pyjedai.datamodel import Data
from pyjedai.workflow import ProgressiveWorkFlow
from pyjedai.utils import (
    values_given,
    get_multiples,
    necessary_dfs_supplied,
    clear_json_file,
    purge_id_column,
    retrieve_top_workflows,
    workflows_to_dataframe)
from pyjedai.block_building import (
    StandardBlocking,
    QGramsBlocking,
    ExtendedQGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking)
                                   
from pyjedai.block_cleaning import (
    BlockFiltering,
    BlockPurging)                         
from pyjedai.comparison_cleaning import (
    WeightedEdgePruning, 
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning, 
    BLAST,
    ReciprocalCardinalityNodePruning,
    ReciprocalWeightedNodePruning,
    ComparisonPropagation)                                   
from pyjedai.prioritization import (
    GlobalTopPM, 
    LocalTopPM, 
    EmbeddingsNNBPM, 
    GlobalPSNM, 
    LocalPSNM, 
    PESM,
    class_references)
from pyjedai.evaluation import Evaluation


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Define arguments with default values
    parser.add_argument('--config',
                        dest='config_name',
                        type=str,
                        default="gt_experiments",
                        help='Name of the experiments config file within script-configs folder (without .json extension)')
    parser.add_argument('--dataset',
                        dest='dataset',
                        type=str,
                        default="d1",
                        help='Which dataset should the gridsearch be conducted for')
    parser.add_argument('--building',
                        dest='allow_building',
                        type=bool,
                        default=False,
                        help='Allow block building')
    parser.add_argument('--filtering',
                        dest='allow_filtering',
                        type=bool,
                        default=False,
                        help='Allow block filtering')
    parser.add_argument('--purging',
                        dest='allow_purging',
                        type=bool,
                        default=False,
                        help='Allow block purging')

    args = parser.parse_args()
    
    #-EDIT-THOSE-#
    # parameters native to the pyjedai progressive workflow
    # don't edit, unless new parameters were added to the workflow
    VALID_WORKFLOW_PARAMETERS = ['matcher',
                                'algorithm',
                                'number_of_nearest_neighbors',
                                'indexing',
                                'similarity_function',
                                'language_model',
                                'tokenizer',
                                'weighting_scheme',
                                'window_size',
                                'qgram']
    # path of the configuration file
    CONFIG_FILE_PATH = to_path('~/pyJedAI/pyJedAI-Dev/script-configs/' + args.config_name + '.json')
    # which configuration from the json file should be used in current experiment  
    EXPERIMENT_NAME = args.config_name + '_' + args.dataset
    # path at which the results will be stored within a json file
    RESULTS_STORE_PATH = to_path('/home/shared/jakub_gpapad/' + EXPERIMENT_NAME)
    JSON_STORE_PATH = RESULTS_STORE_PATH + '.json'
    DF_STORE_PATH = RESULTS_STORE_PATH + '.csv'
    # path at which the top workflows for specified argument values are stored
    BEST_WORKFLOWS_STORE_PATH = to_path(RESULTS_STORE_PATH + '_best.json')
    # results should be stored in the predefined path
    STORE_RESULTS = True
    STORE_RESULTS_AS_DF = True
    # AUC calculation and ROC visualization after execution
    VISUALIZE_RESULTS = True
    # workflow arguments and execution info should be printed in terminal once executed
    PRINT_WORKFLOWS = True
    # identifier column names for source and target datasets
    D1_ID = 'id'
    D2_ID = 'id'
    # methods and their corresponding parameters for MB-based workflows
    # if you don't want to apply filtering, purging or block building (or want to use the default methods when necessary)
    # set those values to None
    _block_building = dict(method=StandardBlocking, 
                            params=dict()) if args.allow_building else None

    _block_filtering = dict(method=BlockFiltering, 
                            params=dict(ratio=0.8)) if args.allow_filtering else None

    _block_purging = dict(method=BlockPurging, 
                            params=dict(smoothing_factor=1.0)) if args.allow_purging else None  
    ##############                 
                                    
    with open(CONFIG_FILE_PATH) as file:
        config = json.load(file)
        
    config = config[args.dataset]
    workflow_config = {k: v for k, v in config.items() if(values_given(config, k) and k in VALID_WORKFLOW_PARAMETERS)}
    workflow_parameters = list(workflow_config.keys())
    workflow_values = list(workflow_config.values())
    workflow_combinations = list(product(*workflow_values))

    if(not necessary_dfs_supplied(config)):
        raise ValueError("Different number of source, target dataset and ground truth paths!")

    datasets_info = list(zip(config['source_dataset_path'], config['target_dataset_path'], config['ground_truth_path']))
    iterations = config['iterations'][0] if(values_given(config, 'iterations')) else 1

    execution_count : int = 0

    if(STORE_RESULTS):
        clear_json_file(path=JSON_STORE_PATH)

    for id, dataset_info in enumerate(datasets_info):
        dataset_id = id + 1
        d1_path, d2_path, gt_path = dataset_info
        dataset_name = config['dataset_name'][id] if(values_given(config, 'dataset_name') and len(config['dataset_name']) > id) else ("D" + str(dataset_id))
        
        sep = config['separator'][id] if values_given(config, 'separator') else '|'
        d1 = pd.read_csv(to_path(d1_path), sep=sep, engine='python', na_filter=False).astype(str)
        d2 = pd.read_csv(to_path(d2_path), sep=sep, engine='python', na_filter=False).astype(str)
        gt = pd.read_csv(to_path(gt_path), sep=sep, engine='python')

        d1_attributes = config['d1_attributes'][id] if values_given(config, 'd1_attributes') else d1.columns.tolist()
        d2_attributes = config['d2_attributes'][id] if values_given(config, 'd2_attributes') else d2.columns.tolist()

        data = Data(
            dataset_1=d1,
            attributes_1=d1_attributes,
            id_column_name_1=D1_ID,
            dataset_2=d2,
            attributes_2=d2_attributes,
            id_column_name_2=D2_ID,
            ground_truth=gt,
        )
        
        true_positives_number = len(gt)
        budgets = config['budget'] if values_given(config, 'budget') else get_multiples(true_positives_number, 10)
        total_workflows = len(workflow_combinations) * len(datasets_info) * len(budgets) * iterations
        
        for budget in budgets:
            for workflow_combination in workflow_combinations:
                workflow_arguments = dict(zip(workflow_parameters, workflow_combination))
                workflow_arguments['budget'] = budget
                workflow_arguments['dataset'] = dataset_name
                
                for iteration in range(iterations):
                    execution_count += 1
                    print(f"#### WORKFLOW {execution_count}/{total_workflows} ####")
                    current_workflow = ProgressiveWorkFlow()
                    current_workflow.run(data=data,
                                        block_building=_block_building,
                                        block_purging=_block_purging,
                                        block_filtering=_block_filtering,
                                        **workflow_arguments)                        
                    current_workflow_info = current_workflow.save(arguments=workflow_arguments,
                                                                  path=JSON_STORE_PATH)
                    if(PRINT_WORKFLOWS):
                        current_workflow.print_info(current_workflow_info)
                
    with open(JSON_STORE_PATH, 'r') as file:
        results = json.load(file)
                
    if(VISUALIZE_RESULTS):
        evaluator = Evaluation(data)
        evaluator.visualize_results_roc(results=results)
        
    if(STORE_RESULTS):
        if(STORE_RESULTS_AS_DF):
            workflows_to_dataframe(workflows=results,
                                store_path=DF_STORE_PATH)
            # os.remove(JSON_STORE_PATH)
        else: 
            with open(JSON_STORE_PATH, 'w', encoding="utf-8") as file:
                json.dump(results, file, indent=4)
    
    
    
                                      

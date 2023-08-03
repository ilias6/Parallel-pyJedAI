import itertools 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import List, Tuple

# Function that creates a confusion matrix
def create_confusion_matrix(confusion_matrix, title):
    
    plt.figure(figsize = (8,5))
    classes = ['Different','Matching']
    cmap = plt.cm.Blues
    plt.grid(False)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],horizontalalignment="center",color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.ylim([1.5, -.5])
    plt.show() 

def plot_feature_progress_per_attribute_group(method_name : str,
                                              dataset_name : str,
                                              feature : str,
                                              attributes : list,
                                              df : pd.DataFrame = None,
                                              load_path : str = None,
                                              grid : bool = True,
                                              save : bool = True,
                                              verbose : bool = True,
                                              in_plot_directory : bool = True
                                              ) -> None:
    """Plots the progress of the value of requested feature per budget for experiments grouped by the attributes.
       Saves the plot as an image in the requested path.
    
    Args:
        method_name (str): Name of the method used in the dataframe's experiments
        dataset_name (str): Name of dataset on which the dataframe's experiments have been applied on
        feature (str): The feature whose per budget progress we want to plot (e.x. auc)
        attributes (list): Group of experiments' arguments whose each distinct combination constitutes a seperate curve
        df (pd.Dataframe): Dataframe containing the information about progressive PER experiments (Defaults to None)
        load_path (str): Path from which the dataframe should be loaded from (Defaults to None)
        grid (bool): Grid to be displayed in the plot (Defaults to True)
        save (bool) : Save the plot as an image on disk (Defaults to True)
        verbose (bool) : Show the produced plot
        in_plot_directory (bool) : Plot to be saved in an experiment directory - 
        created in the target dataframe's / current directory if non-existent (Defaults to True)
    """
    
    experiments : pd.DataFrame
    if(df is not None):
        experiments = df
    elif(load_path is not None):
        experiments = pd.read_csv(load_path)
    else:
        raise ValueError("No dataframe or csv file given - Cannot plot the experiments.")
    
    experiments = experiments.groupby(attributes)


    fig = plt.figure(figsize=(16, 12))
    ax = plt.subplot(111)

    for attributes_unique_values, attributes_experiment_group in experiments:
        group_name = '-'.join([str(attribute) for attribute in attributes_unique_values])
        attributes_experiment_group_per_budget = attributes_experiment_group.sort_values(by='budget').groupby('budget')
        budgets = []
        average_feature_values = []
        for _, current_budget_attributes_experiment_group in attributes_experiment_group_per_budget:
            budgets.append(current_budget_attributes_experiment_group['budget'].mean())
            average_feature_values.append(current_budget_attributes_experiment_group[feature].mean())

        ax.plot(budgets, average_feature_values, label=str(group_name), marker='o', linestyle='-')

    # Customize the plot
    ax.set_title(f'{method_name.capitalize()}/{dataset_name.capitalize()} - Average {feature.capitalize()} vs. Budget Curves')
    ax.set_xlabel('Budget')
    ax.set_ylabel(f'Average {feature.capitalize()}')
    
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(title=attributes, fontsize="9", loc='center right', bbox_to_anchor=(1.23, 0.5))
    
    ax.grid(grid)
    
    if(save):
        file_name = '_'.join([dataset_name, method_name, feature, 'for', '_'.join(attributes)]) + '.png'
        dataframe_directory = os.path.dirname(load_path) if load_path is not None else './'
        store_directory = dataframe_directory if not in_plot_directory else os.path.join(dataframe_directory, 'plots/')        
        
        if in_plot_directory and not os.path.exists(store_directory):
            os.makedirs(store_directory)
            
        plt.savefig(os.path.join(store_directory, file_name))
        
    plt.show()
    
    
    
def plot_attribute_group_avg_ranking(method_name : str,
                                     feature : str,
                                     attributes : list,
                                     dfs : List[pd.DataFrame] = None,
                                     load_paths : List[str] = None,
                                     grid : bool = True,
                                     save : bool = True,
                                     verbose : bool = True,
                                     in_plot_directory : bool = True
                                     ) -> None:
    """For each unique combination of given attributes calculates its average feature value across datasets for each budget.
       Plots the corresponding results and stores them as an image if it is requested.
    
    Args:
        method_name (str): The name of the PER method whose experiments we are evaluating
        feature (str): The feature that we want to evaluate the average ranking of the attribute group for  
        attributes (list): Group of experiments' arguments whose each distinct combination constitutes a seperate curve
        dfs (List[pd.Dataframe]): Dataframes containing the information about progressive PER experiments (Defaults to None)
        load_paths (List[str]): Paths from which the dataframe should be loaded from (Defaults to None)
        grid (bool): Grid to be displayed in the plot (Defaults to True)
        save (bool) : Save the plot as an image on disk (Defaults to True)
        verbose (bool) : Show the produced plot
        in_plot_directory (bool) : Plot to be saved in an experiment directory - 
        created in the target dataframe's / current directory if non-existent (Defaults to True)
    """
    
    if(dfs is None and load_paths is None):
        raise ValueError("No dataframes or csv files given - Cannot calculate and plot average combinations rankings.")

    total_datasets = len(dfs) if dfs is not None else len(load_paths)
    attributes_combinations = {}
    attributes_combinations_budget_scores : List[Tuple[float, str]]
    
    for current_dataset in range(total_datasets):
        if(dfs is not None):
            experiments = dfs[current_dataset]
        else:
            current_dataset_path = load_paths[current_dataset]
            experiments = pd.read_csv(current_dataset_path)
    
        budgets_experiments = experiments.sort_values(by='budget').groupby('budget')

        for current_budget, current_budget_experiments in budgets_experiments:
            current_budget_attributes_combinations = current_budget_experiments.groupby(attributes[0] if len(attributes) == 1 else attributes)
            attributes_combinations_budget_scores = []
            
            for attributes_combination, current_budget_attributes_combination in current_budget_attributes_combinations:
                attributes_combination_budget_feature_value = current_budget_attributes_combination[feature].mean()
                attributes_combinations_budget_scores.append((attributes_combination_budget_feature_value, attributes_combination))
                
            for ranking, attributes_combinations_budget_score in enumerate(sorted(attributes_combinations_budget_scores, reverse=True)):
                attributes_combination_budget_feature_value, attributes_combination = attributes_combinations_budget_score
                if attributes_combination not in attributes_combinations:
                    attributes_combinations[attributes_combination] = {}
                    
                if current_budget not in attributes_combinations[attributes_combination]:
                    attributes_combinations[attributes_combination][current_budget] = []
                    
                attributes_combinations[attributes_combination][current_budget].append(ranking+1)

    fig = plt.figure(figsize=(16, 12))
    ax = plt.subplot(111)       
            
    for attributes_combination, attributes_combination_budgets in attributes_combinations.items():
        
        attributes_combination_average_rankings = []
        sorted_budgets = sorted(attributes_combination_budgets.keys(), reverse=False)
        for budget in sorted_budgets:
            attributes_combination_average_rankings.append(sum(attributes_combination_budgets[budget]) / len(attributes_combination_budgets[budget]))
            
        ax.plot(sorted_budgets, attributes_combination_average_rankings, label=str(attributes_combination), marker='o', linestyle='-')
        
        
    # Customize the plot
    ax.set_title(f'{method_name.capitalize()} - Average {feature.capitalize()} Ranking vs. Budget Curves')
    ax.set_xlabel('Budget')
    ax.set_ylabel('Average Ranking')
    
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(title=attributes, fontsize="9", loc='center right', bbox_to_anchor=(1.23, 0.5))
    
    ax.grid(grid)
    
    if(save):
        file_name = '_'.join([method_name, 'for', '_'.join(attributes), 'avg_rankings', feature]) + '.png'
        dataframe_directory = os.path.dirname(load_paths[0]) if load_paths is not None else './'
        store_directory = dataframe_directory if not in_plot_directory else os.path.join(dataframe_directory, 'avg_rankings/')        
        
        if in_plot_directory and not os.path.exists(store_directory):
            os.makedirs(store_directory)
            
        plt.savefig(os.path.join(store_directory, file_name))
        
    plt.show()
    
    
    
def plot_attribute_group_avg_top_distance(method_name : str,
                                     feature : str,
                                     attributes : list,
                                     dfs : List[pd.DataFrame] = None,
                                     load_paths : List[str] = None,
                                     grid : bool = True,
                                     save : bool = True,
                                     verbose : bool = True,
                                     in_plot_directory : bool = True
                                     ) -> None:
    """For each unique combination of given attributes calculates its feature's value average difference from the best value across datasets for each budget.
       Plots the corresponding results and stores them as an image if it is requested.
    
    Args:
        method_name (str): The name of the PER method whose experiments we are evaluating
        feature (str): The feature that we want to evaluate the average ranking of the attribute group for  
        attributes (list): Group of experiments' arguments whose each distinct combination constitutes a seperate curve
        dfs (List[pd.Dataframe]): Dataframes containing the information about progressive PER experiments (Defaults to None)
        load_paths (List[str]): Paths from which the dataframe should be loaded from (Defaults to None)
        grid (bool): Grid to be displayed in the plot (Defaults to True)
        save (bool) : Save the plot as an image on disk (Defaults to True)
        verbose (bool) : Show the produced plot
        in_plot_directory (bool) : Plot to be saved in an experiment directory - 
        created in the target dataframe's / current directory if non-existent (Defaults to True)
    """
    
    if(dfs is None and load_paths is None):
        raise ValueError("No dataframes or csv files given - Cannot calculate and plot average combinations rankings.")

    total_datasets = len(dfs) if dfs is not None else len(load_paths)
    attributes_combinations = {}
    attributes_combinations_budget_scores : List[Tuple[float, str]]
    
    for current_dataset in range(total_datasets):
        if(dfs is not None):
            experiments = dfs[current_dataset]
        else:
            current_dataset_path = load_paths[current_dataset]
            experiments = pd.read_csv(current_dataset_path)
    
        budgets_experiments = experiments.sort_values(by='budget').groupby('budget')

        for current_budget, current_budget_experiments in budgets_experiments:
            current_budget_attributes_combinations = current_budget_experiments.groupby(attributes[0] if len(attributes) == 1 else attributes)
            attributes_combinations_budget_scores = []
            
            for attributes_combination, current_budget_attributes_combination in current_budget_attributes_combinations:
                attributes_combination_budget_feature_value = current_budget_attributes_combination[feature].mean()
                attributes_combinations_budget_scores.append((attributes_combination_budget_feature_value, attributes_combination))
                
            attributes_combinations_budget_scores = sorted(attributes_combinations_budget_scores, reverse=True)
            budget_highest_feature_value = attributes_combinations_budget_scores[0][0]
                
            for attributes_combinations_budget_score in attributes_combinations_budget_scores:
                attributes_combination_budget_feature_value, attributes_combination = attributes_combinations_budget_score
                if attributes_combination not in attributes_combinations:
                    attributes_combinations[attributes_combination] = {}
                    
                if current_budget not in attributes_combinations[attributes_combination]:
                    attributes_combinations[attributes_combination][current_budget] = []
                    
                attributes_combinations[attributes_combination][current_budget].append(budget_highest_feature_value - attributes_combination_budget_feature_value)

    fig = plt.figure(figsize=(16, 12))
    ax = plt.subplot(111)       
            
    for attributes_combination, attributes_combination_budgets in attributes_combinations.items():
        
        attributes_combination_average_rankings = []
        sorted_budgets = sorted(attributes_combination_budgets.keys(), reverse=False)
        for budget in sorted_budgets:
            attributes_combination_average_rankings.append(sum(attributes_combination_budgets[budget]) / len(attributes_combination_budgets[budget]))
            
        ax.plot(sorted_budgets, attributes_combination_average_rankings, label=str(attributes_combination), marker='o', linestyle='-')
        
        
    # Customize the plot
    ax.set_title(f'{method_name.capitalize()} - Average {feature.capitalize()} Distance from Top vs. Budget Curves')
    ax.set_xlabel('Budget')
    ax.set_ylabel('Average Distance from Top')
    
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(title=attributes, fontsize="9", loc='center right', bbox_to_anchor=(1.23, 0.5))
    
    ax.grid(grid)
    
    if(save):
        file_name = '_'.join([method_name, 'for', '_'.join(attributes), 'avg_distances', feature]) + '.png'
        dataframe_directory = os.path.dirname(load_paths[0]) if load_paths is not None else './'
        store_directory = dataframe_directory if not in_plot_directory else os.path.join(dataframe_directory, 'avg_distances/')        
        
        if in_plot_directory and not os.path.exists(store_directory):
            os.makedirs(store_directory)
            
        plt.savefig(os.path.join(store_directory, file_name))
        
    plt.show()
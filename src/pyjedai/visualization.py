import itertools 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

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
    feature_acronyms = {
        "algorithm": "alg",
        "number_of_nearest_neighbors": "nnn",
        "indexing": "ind",
        "similarity_function": "sf",
        "language_model": "lm",
        "tokenizer": "tkn",
        "weighting_scheme": "ws",
        "window_size": "wsz",    
    }
    
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
    
    
    
#     # Create some sample data
# data = np.arange(20)

# # Create a matplotlib figure
# fig, ax = plt.subplots()

# # Create multiple plots 
# for i in range(7):
#     ax.plot(data, i * data, label=f'y={i}x')

# # Set title and labels
# ax.set_title('Example plot')
# ax.set_xlabel('x')
# ax.set_ylabel('y')

# # Add a legend
# pos = ax.get_position()
# ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
# ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
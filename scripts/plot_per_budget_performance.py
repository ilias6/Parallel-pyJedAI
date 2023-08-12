from pyjedai.visualization import plot_attributes_performance_for_budget
import os

def valid_file(file : str):
    return ('experiments' in file and '.csv' in file)

def get_attributes(file : str):
    if 'pesm' in file or 'gt' in file or 'test' in file:
        return ["weighting_scheme", "algorithm"]
    elif 'gsn' in file or 'lsn' in file:
        return ["weighting_scheme", "window_size", "algorithm"]
    elif 'nn' in file:
        return ["number_of_nearest_neighbors", "language_model", "similarity_function"]
    elif 'join' in file:
        return ["number_of_nearest_neighbors", "similarity_function", "tokenizer", "weighting_scheme", "qgram"]
    else:
        return ["indexing"]
    
def get_method_csv_files_for_directory(method : str, directory_path: str) -> list:
    file_names = os.listdir(directory_path)
    return sorted([(directory_path + file_name) for file_name in file_names if (valid_file(file_name) and method in file_name)])

# path where the experiment results are stored
EXPERIMENTS_PATH = "/home/shared/jakub_gpapad/"
FEATURES = ['auc', 'recall', 'time']
CALCULATE_DISTANCES = [True, False]

# substring of each file's name till the first underscore character corresponds to its method
# we gather all the unique methods for which the calculations have concluded
file_list = os.listdir(EXPERIMENTS_PATH)
methods = set([file_name.split('_')[0] for file_name in file_list if valid_file(file_name)])
total_files = len(methods) * len(FEATURES) * len(CALCULATE_DISTANCES)
current_file = 0

for method in methods:
    attributes = get_attributes(method)
    for feature in FEATURES:
        for calculate_distance in CALCULATE_DISTANCES:
            current_file += 1
            metric = "distance" if calculate_distance else "ranking"
            print(f"{current_file}/{total_files} : Method[{method}] Attributes{attributes} Feature[{feature}] Metric[{metric}]")
            plot_attributes_performance_for_budget(
                                                    method_name = method,
                                                    feature = feature,
                                                    attributes = attributes,
                                                    load_paths = get_method_csv_files_for_directory(method=method, directory_path=EXPERIMENTS_PATH),
                                                    calculate_distance=calculate_distance
                                                  )
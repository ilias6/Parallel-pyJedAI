from pyjedai.visualization import plot_feature_progress_per_attribute_group
import os

def valid_file(file : str):
    return ('experiments' in file and '.csv' in file)

def get_attributes(file : str):
    if 'pesm' in file or 'gt' in file:
        return ["weighting_scheme", "algorithm"]
    elif 'gsn' in file or 'lsn' in file:
        return ["weighting_scheme", "window_size", "algorithm"]
    else:
        return ["indexing"]

# path where the experiment results are stored
EXPERIMENTS_PATH = "/home/shared/jakub_gpapad/"
# Get a list of all files in the directory
file_list = os.listdir(EXPERIMENTS_PATH)

# Filter the files that contain 'experiments' in their name
valid_files = [file_name for file_name in file_list if valid_file(file_name)]
features = ['auc', 'recall', 'time']
total_files = len(valid_files) * len(features)
current_file = 0

for valid_file in valid_files:
    load_path = EXPERIMENTS_PATH + valid_file
    attributes = get_attributes(valid_file)
    file_info = valid_file.replace('experiments', '').('vector', '').replace('.csv', '').split('_')
    method_name = file_info[0]
    dataset_name = file_info[-1]
    
    for feature in features:
        current_file += 1
        print(f"{current_file}/{total_files} : Name[{valid_file}] Method[{method_name}] Dataset[{dataset_name}] Feature[{feature}]")
        plot_feature_progress_per_attribute_group(
            load_path=load_path,
            method_name=method_name,
            dataset_name=dataset_name,
            feature=feature,
            attributes = attributes
        )
        

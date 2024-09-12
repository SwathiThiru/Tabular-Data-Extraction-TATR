import os
import csv
import yaml
from tabulate import tabulate
from sklearn.metrics import precision_score, recall_score

def read_ground_truth(file_path):
    with open(file_path, 'r') as yaml_file:
        ground_truth = yaml.safe_load(yaml_file)
    return ground_truth

def read_predicted_data(file_path):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        predicted_data = [row for row in reader]
    return predicted_data

def flatten_nested_list(nested_list):
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]

'''def flatten_dict_items(dictionary):
    """Flatten a dictionary into a list of interleaved keys and values."""
    return [item for pair in dictionary.items() for item in pair]

def flatten_ground_truth(ground_truth):
    """Flatten a dictionary of lists into a single list."""
    flattened_list = []
    for key, values in ground_truth.items():
        flattened_list.append(key)
        flattened_list.extend(values)
    return flattened_list'''

def flatten_dict_with_lists(input_dict):
    """Flatten a dictionary with alternating keys and values."""
    flattened_list = []
    for key, value in input_dict.items():
        flattened_list.append(key)
        if isinstance(value, list):
            flattened_list.extend(value)
        else:
            flattened_list.append(value)
    return flattened_list

def evaluate_table_extraction(ground_truth, predicted_data):
    '''if isinstance(ground_truth, dict):
        # Flatten ground truth dictionary into a list of interleaved keys and values
        ground_truth_flat = flatten_dict_items(ground_truth)
        #ground_truth_flat = flatten_ground_truth(ground_truth)
    else:
        # Use ground truth as-is if it's already a list
        ground_truth_flat = flatten_nested_list(ground_truth)'''

    # Flatten ground truth dictionary into a list of interleaved keys and values
    ground_truth_flat = flatten_dict_with_lists(ground_truth)
    # Flatten predicted data
    predicted_data_flat = [item.replace('|', '').strip() for item in flatten_nested_list(predicted_data)]

    # Add a conditional step to pad predicted_data_flat if needed
    if len(ground_truth_flat) > len(predicted_data_flat):
        padding_size = len(ground_truth_flat) - len(predicted_data_flat)
        predicted_data_flat.extend([''] * padding_size)
    elif len(ground_truth_flat) < len(predicted_data_flat):
        padding_size = len(predicted_data_flat) - len(ground_truth_flat)
        ground_truth_flat.extend([''] * padding_size)

    # Compute precision and recall
    precision = precision_score(ground_truth_flat, predicted_data_flat, average='micro')
    recall = recall_score(ground_truth_flat, predicted_data_flat, average='micro')

    return precision, recall

def evaluate_folder(folder_path, ground_truth_folder):
    precision_sum = 0
    recall_sum = 0
    num_files = 0

    for csv_file in os.listdir(folder_path):
        if csv_file.endswith(".csv"):
            #csv_path = os.path.join(folder_path, csv_file)
            csv_path = folder_path + '/' + csv_file
            yaml_file = ground_truth_folder + '/' + os.path.splitext(csv_file)[0] + ".yml"
            #yaml_file = os.path.join(folder_path, os.path.splitext(csv_file)[0] + ".yaml")

            if os.path.exists(yaml_file):
                ground_truth = read_ground_truth(yaml_file)
                predicted_data = read_predicted_data(csv_path)

                precision, recall = evaluate_table_extraction(ground_truth, predicted_data)

                print(f'File: {csv_file}, Precision: {precision:.4f}, Recall: {recall:.4f}')

                precision_sum += precision
                recall_sum += recall
                num_files += 1

    print('Number of files evaluated : ', num_files)

    if num_files > 0:
        average_precision = precision_sum / num_files
        average_recall = recall_sum / num_files
        print(f'Average Precision: {average_precision:.4f}')
        print(f'Average Recall: {average_recall:.4f}')

def main():
    # Replace with your folder paths
    predicted_folder = '../inferences/extractionOutput'
    ground_truth_folder = '../inferences/groundTruth'

    print("Evaluation results for each file:")
    evaluate_folder(predicted_folder, ground_truth_folder)

if __name__ == "__main__":
    main()

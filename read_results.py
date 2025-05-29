import os
import json
import numpy as np
import pandas as pd
from tabulate import tabulate

def process_dataset(dataset_name):
    base_dir = 'temp'
    dataset_dir = os.path.join(base_dir, dataset_name)

    results = []
    k = 60

    for model in ['SA', 'LLY', 'GL', 'SEQL', 'SAND']:
        test_accs = []

        for seed in range(1, 11):
            results_dir = os.path.join(dataset_dir, f'{k}/{model}_seed_{seed}', 'fit')
            results_file = os.path.join(results_dir, 'results.json')

            with open(results_file, 'r') as f:
                result = json.load(f)
                test_accs.append(result['test'])

        test_mean, test_std = np.mean(test_accs), np.std(test_accs)

        results.append({'Dataset': dataset_name, 'Model': model, 'Test Mean': test_mean, 'Test Std': test_std})

    return results

# List of datasets
datasets = ['mice', 'mnist', 'fashion', 'isolet', 'coil', 'activity']

# Process each dataset and append the results to the list
all_results = []
previous_dataset = None

for dataset in datasets:
    results = process_dataset(dataset)
    if previous_dataset is not None and previous_dataset != dataset:
        all_results.append({key: '---' for key in results[0].keys()})
    all_results.extend(results)
    previous_dataset = dataset

# Create a DataFrame from the list of results
df = pd.DataFrame(all_results)

# Round numerical columns to 4 decimal places
numeric_cols = ['Test Mean', 'Test Std']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').round(3)
df = df.fillna('---')

# Combine mean and std columns
df['Test'] = df.apply(lambda row: f"{row['Test Mean']} ± {row['Test Std']}", axis=1)

# Drop the original mean and std columns
df = df.drop(columns=['Test Mean', 'Test Std'])


# Print the DataFrame
print(tabulate(df, headers='keys', tablefmt='psql'))

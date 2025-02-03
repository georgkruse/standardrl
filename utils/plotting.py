import yaml
import os
import ray
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import numpy as np
import re

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot_single_run(path):
    # Load the results.json file
    results_path = os.path.join(path, 'results.json')
    
    results = pd.read_json(results_path, lines=True)
    # Filter out rows where 'episodic_return' exists as a key and ensure only single 'episodic_return' entries are considered

    results = results[results.apply(lambda row: 'episodic_return' in row and not isinstance(row['episodic_return'], type(np.nan)), axis=0)]
    # Extract episode_return and global_step
    episode_returns = results['episodic_return'].values
    global_steps = results['global_step'].values  

    # Plot episode_return vs. global_step
    plt.figure(figsize=(10, 5))
    plt.plot(global_steps, episode_returns, label='Episode Return')
    plt.xlabel('Global Step')
    plt.ylabel('Episode Return')
    plt.title('Episode Return vs. Global Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, 'training_results.png'))

def plot_tune_run(path):
    fig, axis = plt.subplots(figsize=(8,7))

    paths, labels, types = [], [], []
    paths += [path]

    title = '4 qubits local grid 1 env'
    fig_name = f'{title}'
    data_length = 300


    labels += [["0.1",
                "0.05",
                "0.01",
                "0.005",
                "0.001"
            ]]

    types += [{'0': ['=0.1'],
            '1': ['=0.05'],
            '2': ['=0.01'],
            '3': ['=0.005'],
            '4': ['=0.001']
            }]

    for idx, path in enumerate(paths):
        # path = os.path.basename(os.path.normpath(path))
        curves = [[] for _ in range(len(labels[idx]))]
        curves_x = [[] for _ in range(len(labels[idx]))]


        results_file_name = "/result.json"
        result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
        results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]
        result = pd.concat(results)
        
        for key, value in types[idx].items():
            for i, data_exp in enumerate(results):
                if all(x in result_files[i] for x in value):
                    curves_x[int(key)].append(data_exp['episode'].values[:data_length])
                    curves[int(key)].append(data_exp['episodic_return'].values[:data_length])


        for id, curve in enumerate(curves):
            data = np.vstack(curve)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            upper = mean +  std
            lower = mean  -  std
            axis.plot(data_exp['episode'].values[:data_length], mean, label=labels[idx][id])
            axis.fill_between(data_exp['episode'].values[:data_length], lower, upper, alpha=0.5)



    axis.set_xlabel("$Episodes$", fontsize=13)
    axis.set_ylabel("$Return$", fontsize=15)
    axis.set_title(title, fontsize=15)
    axis.legend(fontsize=12, loc='lower left')
    axis.minorticks_on()
    axis.grid(which='both', alpha=0.4)
    fig.tight_layout()
    fig.savefig(f'{fig_name}.png', dpi=100)


if __name__ == "__main__":
    path = 'H:\\standardrl\\logs\\2025-02-03--14-33-26_RL_'
    # path = 'H:\\standardrl\\logs\\2025-02-03--14-18-00_QRL_\\train_agent_2025-02-03_14-18-00\\'
    plot_single_run(path)
    # plot_tune_run(path)

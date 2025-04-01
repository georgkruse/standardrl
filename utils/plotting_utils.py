import yaml
import os
import ray
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot_single_run(path):
    # Load the results.json file
    results_path = os.path.join(path, 'result.json')
    
    results = pd.read_json(results_path, lines=True)

    results_selected_returns = results[pd.notna(results['episodic_return'])]
    episode_returns = results_selected_returns['episodic_return'].values
    global_steps_rewards = results_selected_returns['global_step'].values

    results_selected_loss = results[pd.notna(results['loss'])]
    episode_losses = results_selected_loss['loss'].values
    global_steps_loss = results_selected_loss['global_step'].values

    # Plot episode_return vs. global_step
    plt.figure(figsize=(10, 5))
    plt.plot(global_steps_rewards, episode_returns, label='Episode Return')
    plt.xlabel('Global Step')
    plt.ylabel('Episode Return')
    plt.title('Episode Return vs. Global Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, 'training_results_reward.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(global_steps_loss, episode_losses, label='Episode Loss')
    plt.xlabel('Global Step')
    plt.ylabel('Episode Loss')
    plt.title('Episode Return vs. Global Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, 'training_results_loss.png'))



def plot_tune_run(path):
    fig, axis = plt.subplots(figsize=(8,7))

    paths, labels, types = [], [], []
    paths += [path]

    title = 'CartPole Demo'
    fig_name = f'{title}'

    labels += [["lr = 0.01",
                "lr = 0.001"]]

    types += [{'0': ['learning_rate=0.01'],
               '1': ['learning_rate=0.001']}]

    for idx, path in enumerate(paths):
        curves = [[] for _ in range(len(labels[idx]))]
        curves_x = [[] for _ in range(len(labels[idx]))]

        # Get subdirectories and result files
        subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        result_files = []
        for subdir in subdirs:
            result_files.extend([os.path.join(subdir, f) for f in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, f))])
        
        # Load JSON results
        results = [pd.read_json(f + os.sep + 'result.json', lines=True) for f in result_files]
        
        # Group data by type
        for key, value in types[idx].items():
            for i, data_exp in enumerate(results):
                if all(x in result_files[i] for x in value):
                    curves_x[int(key)].append(data_exp['global_step'].values)
                    curves[int(key)].append(data_exp['episodic_return'].values)

        # Process each curve
        for id, (curve, curve_x) in enumerate(zip(curves, curves_x)):
            if not curve:  # Skip if no data
                continue

            # Define a common global_step range
            min_step = min(x.min() for x in curve_x)
            max_step = max(x.max() for x in curve_x)
            common_steps = np.linspace(min_step, max_step, num=300)  # Adjust num for resolution

            # Interpolate each run to the common steps
            interpolated_returns = []
            for x, y in zip(curve_x, curve):
                # Remove NaNs if present
                mask = ~np.isnan(y)
                if mask.sum() < 2:  # Need at least 2 points for interpolation
                    continue
                x_clean, y_clean = x[mask], y[mask]
                interp_func = interp1d(x_clean, y_clean, bounds_error=False, fill_value="extrapolate")
                interpolated_returns.append(interp_func(common_steps))
            
            if not interpolated_returns:
                continue

            # Compute mean and std across interpolated runs
            data = np.vstack(interpolated_returns)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            upper = mean + std
            lower = mean - std

            # Plot
            axis.plot(common_steps, mean, label=labels[idx][id])
            axis.fill_between(common_steps, lower, upper, alpha=0.5)

    # Customize plot
    axis.set_xlabel("Global Step", fontsize=13)
    axis.set_ylabel("Return", fontsize=15)
    axis.set_title(title, fontsize=15)
    axis.legend(fontsize=12, loc='lower left')
    axis.minorticks_on()
    axis.grid(which='both', alpha=0.4)
    fig.tight_layout()
    fig.savefig(f'{path}{os.sep}{fig_name}.png', dpi=100)
    plt.close(fig)  # Close the figure to free memory

if __name__ == "__main__":
    # path = 'H:\\standardrl\\logs\\2025-02-03--14-18-00_QRL_\\train_agent_2025-02-03_14-18-00\\'
    path = 'H:\\standardrl\\logs\\2025-04-01--09-27-26_RL'
    # plot_single_run(path)
    plot_tune_run(path)
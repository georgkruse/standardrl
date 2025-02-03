import argparse
import yaml
from collections import namedtuple
import ray
import os
import datetime
import shutil
from utils.generate_config import generate_config
from utils.train_function import train_agent
import json
import pandas as pd

if __name__ == "__main__":

    config_path = 'configs/ppo_default.yaml'

    with open(config_path) as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())

    param_space = generate_config(config)
    
    name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config.method + '_'
    ray_path = os.getcwd() + '/' + config.ray_logging_path
    path = ray_path + "/" + name
    os.makedirs(os.path.dirname(path + '/'), exist_ok=True)
    shutil.copy(config_path, path + '/alg_config.yml')
    param_space['path'] = path
    # ray.init(_temp_dir=os.getcwd() + os.sep +  'logs')

    # ray.init(log_to_driver=True, _temp_dir=path)
    train_agent(param_space)
    import matplotlib.pyplot as plt

    # Load the results.json file
    results_path = os.path.join(path, 'results.json')
    
    results = pd.read_json(results_path, lines=True)

    # Extract episode_return and global_step
    episode_returns = [entry['episode_return'] for entry in results]
    global_steps = [entry['global_step'] for entry in results]

    # Plot episode_return vs. global_step
    plt.figure(figsize=(10, 5))
    plt.plot(global_steps, episode_returns, label='Episode Return')
    plt.xlabel('Global Step')
    plt.ylabel('Episode Return')
    plt.title('Episode Return vs. Global Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, 'training_results.png'))





import yaml
from collections import namedtuple
import os
import datetime
import shutil
from utils.generate_config import generate_config
from utils.train_function import train_agent
from utils.plotting import plot_single_run


if __name__ == "__main__":

    config_path = f'configs{os.sep}dqn_default.yaml'

    with open(config_path) as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())

    param_space = generate_config(config)
    
    name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config.method + '_'
    ray_path = os.getcwd() + os.sep + config.ray_logging_path
    path = ray_path + os.sep + name
    param_space['path'] = path

    os.makedirs(os.path.dirname(path + os.sep), exist_ok=True)
    shutil.copy(config_path, path + os.sep + 'alg_config.yml')

    train_agent(param_space)
    plot_single_run(path)






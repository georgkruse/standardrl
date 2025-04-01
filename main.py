import os
import yaml
import datetime
import shutil
from utils.config_utils import generate_config
from utils.train_utils import train_agent
from utils.plotting_utils import plot_single_run


if __name__ == "__main__":

    config_path = f'configs{os.sep}dqn_default.yaml'
    # Specify the path to the config file
    config_path = 'configs/dqn_default.yaml'

    # Load the config file 
    with open(config_path) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    # Generate the parameter space for the experiment from the config file
    parameter_config = generate_config(config['algorithm_config'])
    tune_config = config['tune_config']
    
    name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config.method + '_'
    ray_path = os.getcwd() + os.sep + config.ray_logging_path
    path = ray_path + os.sep + name

    os.makedirs(os.path.dirname(path + os.sep), exist_ok=True)
    shutil.copy(config_path, path + os.sep + 'alg_config.yml')

    # Start the agent training 
    train_agent(parameter_config)
    # Plot the results of the training
    plot_single_run(path)






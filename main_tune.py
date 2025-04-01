import os
import ray
from ray import tune
import yaml
import datetime
import shutil
from utils.config_utils import generate_config
from utils.train_utils import train_agent
from utils.plotting_utils import plot_tune_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "This parser receives the yaml config file")
    parser.add_argument("--config", default = "configs/dqn_default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())

    ray.init(local_mode = config.ray_local_mode,
             num_cpus = config.num_cpus,
             num_gpus=config.num_gpus,
    # Specify the path to the config file
    config_path = 'configs/dqn_default.yaml'

    # Load the config file 
    with open(config_path) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    # Generate the parameter space for the experiment from the config file
    parameter_config = generate_config(config['algorithm_config'])
    tune_config = config['tune_config']
    
    # Based on the current time, create a unique name for the experiment
    name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + tune_config['trial_name']
    path = os.path.join(os.getcwd(), tune_config['trial_path'], name)
    parameter_config['path'] = path

    # Create the directory and save a copy of the config file so 
    # that the experiment can be replicated
    os.makedirs(os.path.dirname(path + '/'), exist_ok=True)
    shutil.copy(config_path, path + '/config.yml')

    # Instead of running a single agent as before, we will use ray.tune to run multiple agents
    # in parallel. We will use the same train_agent function as before.
    ray.init(local_mode = tune_config['ray_local_mode'],
             num_cpus = tune_config['num_cpus'],
             num_gpus= tune_config['num_gpus'],
             _temp_dir=os.path.join(os.getcwd(), 'logs', 'tmp_ray_logs'),
             include_dashboard = False)
    
    param_space = generate_config(config)
    
    name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config.method 
    path = os.path.join(os.getcwd(), config.ray_logging_path, name)

    os.makedirs(os.path.dirname(path + os.sep), exist_ok=True)
    shutil.copy(args.config, path + os.sep +'alg_config.yml')

    def trial_name_creator(trial):
        return trial.__str__() + '_' + trial.experiment_tag + ','
    
    # We will use the tune.Tuner class to run multiple agents in parallel
    tuner = tune.Tuner(
            train_agent,
            tune_config=tune.TuneConfig(num_samples=tune_config['ray_num_trial_samples'],
                                        trial_dirname_creator=trial_name_creator),
            run_config=tune.RunConfig(storage_path=path),
            param_space=parameter_config,
            )
    # The fit function will start the hyperparameter search
    tuner.fit()

    # After the experiment is done, we will plot the results.
    ray.shutdown()

    plot_tune_run(path)





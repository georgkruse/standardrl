import argparse
import yaml
from collections import namedtuple
import ray
import os
import datetime
import shutil
from ray import tune
from utils.generate_config import generate_config
from utils.train_function import train_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "This parser receives the yaml config file")
    parser.add_argument("--config", default = "configs/dqn_default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())
    path = None

    ray.init(local_mode = config.ray_local_mode,
             num_cpus = config.num_cpus,
             num_gpus=config.num_gpus,
             _temp_dir=os.path.join(os.getcwd(),'ray_logs'),
             include_dashboard = False)
    
    param_space = generate_config(config)
    
    name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config.method + '_'
    ray_path = os.getcwd() + '/' + config.ray_logging_path
    path = ray_path + "/" + name

    os.makedirs(os.path.dirname(path + '/'), exist_ok=True)
    shutil.copy(args.config, path + '/alg_config.yml')

    def trial_name_creator(trial):
            return trial.__str__() + '_' + trial.experiment_tag + ','
    
    tuner = tune.Tuner(
            train_agent,
            tune_config=tune.TuneConfig(num_samples=config.ray_num_trial_samples,
                                        trial_dirname_creator=trial_name_creator),
            param_space=param_space,)
        
    tuner.fit()
    ray.shutdown()





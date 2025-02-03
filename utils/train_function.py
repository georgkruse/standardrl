from agents.ppo import ppo
from agents.dqn import dqn
from agents.reinforce import reinforce

agent_switch = {
    "PPO": ppo, # discrete PPO
    "DQN": dqn,
    "REINFORCE": reinforce
}

def train_agent(config):
    agent_switch[config["agent"]](config)

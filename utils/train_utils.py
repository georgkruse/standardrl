from agents.ppo import ppo
from agents.dqn import dqn
from agents.reinforce import reinforce

# Here we specify all algorithms which we want to use in our experiments
# To add a new agent, simply add a new entry to the dictionary and import the agent
agent_switch = {
    "PPO": ppo, 
    "DQN": dqn,
    "REINFORCE": reinforce
}

def train_agent(config):
    agent_switch[config["agent"]](config)

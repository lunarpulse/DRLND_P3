from unityagents import UnityEnvironment
import numpy as np

from agent import MADDPG
from model import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim

env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agents = MADDPG(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=0)

agents.agents[0].actor_local.load_state_dict(torch.load('./tennis_checkpoint_actor_0.pth'))
agents.agents[0].critic_local.load_state_dict(torch.load('./tennis_checkpoint_critic_0.pth'))
agents.agents[1].actor_local.load_state_dict(torch.load('./tennis_checkpoint_actor_1.pth'))
agents.agents[1].critic_local.load_state_dict(torch.load('./tennis_checkpoint_critic_1.pth'))

env_info = env.reset(train_mode=False)[brain_name]        
states = env_info.vector_observations                  
scores = np.zeros(num_agents)                          

while True:
    actions = agents.act(states, add_noise=False)                    
    env_info = env.step(actions)[brain_name]        
    next_states = env_info.vector_observations        
    rewards = env_info.rewards                        
    dones = env_info.local_done                 
    scores += rewards                         
    states = next_states                              
    if np.any(dones):
        print ("Final score {}".format(scores))
        break

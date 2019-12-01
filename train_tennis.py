from unityagents import UnityEnvironment
import numpy as np

from agent import MADDPG
import torch
import torch.optim as optim

from collections import deque
import time
import matplotlib.pyplot as plt

def maddpg(n_episodes=2000, max_t=1000, print_every=10):
    t0=time.time()
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = np.zeros(num_agents)
        agents.reset() # noise reset only currently
        for t in range(max_t):
            action = agents.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += rewards
            agents.step(state, action, rewards, next_state, dones)
            state = next_state
            if np.any(dones):
                break 
        score = np.max(score)
        scores.append(score)
        scores_deque.append(np.mean(score))
        average_score = np.mean(scores_deque)

        if i_episode > 0 and i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, average_score))
        if average_score > 0.5:
            print("\nProblem Solved!")
            break
    t1=time.time()

    print("\nTotal time elapsed: {} seconds".format(t1-t0))
    torch.save(agents.agents[0].actor_local.state_dict(), 'tennis_checkpoint_actor_0.pth')
    torch.save(agents.agents[0].critic_local.state_dict(), 'tennis_checkpoint_critic_0.pth') 
    torch.save(agents.agents[1].actor_local.state_dict(), 'tennis_checkpoint_actor_1.pth')
    torch.save(agents.agents[1].critic_local.state_dict(), 'tennis_checkpoint_critic_1.pth')
    
    return scores

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

#initialisation of the model
agents = MADDPG(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=10)

n_episodes = 10000
print_every = 50

#training for n_episodes
score = maddpg(n_episodes=n_episodes, print_every=print_every)

# Plot the result
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(score)+1), score)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
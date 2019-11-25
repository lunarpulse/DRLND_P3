#!/usr/bin/env python

from unityagents import UnityEnvironment
import numpy as np


env = UnityEnvironment(file_name="/home/lunarpulse/Documents/DRLND/deep-reinforcement-learning/p3_collab-compet/Soccer_Linux/Soccer.x86_64")

# print the brain names
print(env.brain_names)

# set the goalie brain
g_brain_name = env.brain_names[0]
g_brain = env.brains[g_brain_name]

# set the striker brain
s_brain_name = env.brain_names[1]
s_brain = env.brains[s_brain_name]


# reset the environment
env_info = env.reset(train_mode=True)

# number of agents 
num_g_agents = len(env_info[g_brain_name].agents)
print('Number of goalie agents:', num_g_agents)
num_s_agents = len(env_info[s_brain_name].agents)
print('Number of striker agents:', num_s_agents)

# number of actions
g_action_size = g_brain.vector_action_space_size
print('Number of goalie actions:', g_action_size)
s_action_size = s_brain.vector_action_space_size
print('Number of striker actions:', s_action_size)

# examine the state space 
g_states = env_info[g_brain_name].vector_observations
g_state_size = g_states.shape[1]
print('There are {} goalie agents. Each receives a state with length: {}'.format(g_states.shape[0], g_state_size))
s_states = env_info[s_brain_name].vector_observations
s_state_size = s_states.shape[1]
print('There are {} striker agents. Each receives a state with length: {}'.format(s_states.shape[0], s_state_size))


from agent import DDPG_agent
from model import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay


class MADDPG:
    def __init__(self, in_actor, in_critic, action_size, num_agents, random_seed):
        self.agents = [DDPG_agent(in_actor, in_critic, action_size, num_agents, random_seed), 
                      DDPG_agent(in_actor, in_critic, action_size, num_agents, random_seed)]
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.num_agents = num_agents
        
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = [agent.act(state, add_noise) for agent, state in zip(self.agents, states)]
        return actions
    
    def target_act(self, states):
        """Returns actions for given state as per current policy."""
        actions = [agent.target_act(state) for agent, state in zip(self.agents, states)]
        return actions
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        #for i in range(state.shape[0]):
        state = np.asanyarray(state)
        action = np.asanyarray(action)
        reward = np.asanyarray(reward)
        next_state = np.asanyarray(next_state)
        done = np.asanyarray(done)
        self.memory.add(state.reshape((1, self.num_agents, -1)), action.reshape((1, self.num_agents, -1)),                         reward.reshape((1, self.num_agents, -1)), next_state.reshape((1,self.num_agents, -1)),                         done.reshape((1, self.num_agents, -1)))
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for ai in range(self.num_agents):
                experiences = self.memory.sample()
                self.learn(experiences, ai, GAMMA)
    
    def reset(self):
        #print("Agents {}".format(self.agents[0]))
        #self.agents[0].reset()
        [agent.reset() for agent in self.agents]
        
    def learn(self, experiences, ai, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences

        agent = self.agents[ai]
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        next_states = next_states.view(1, BATCH_SIZE, self.num_agents, -1)
        actions_next = self.target_act(next_states)
        actions_next = torch.cat(actions_next, dim=1)
        next_states = next_states.view(BATCH_SIZE,-1)
        actions_next = actions_next.view(BATCH_SIZE,-1)
        #print (actions_next.shape)
        #print (next_states.shape)
        #print (next_states.shape)
        #print (actions_next.shape)
        #print( actions_next[0] )
        
        Q_targets_next = agent.critic_target(next_states, actions_next)
        #print (rewards[:,ai].shape)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards[:,ai] + (gamma * Q_targets_next * (1 - dones[:,ai]))
        # Compute critic loss
        Q_expected = agent.critic_local(states.view(BATCH_SIZE,-1), actions.view(BATCH_SIZE,-1))
        # mean squared error loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        # zero_grad because we do not want to accumulate 
        # gradients from other batches, so needs to be cleared
        agent.critic_optimizer.zero_grad()
        # compute derivatives for all variables that
        # requires_grad-True
        critic_loss.backward()
        # update those variables that requires_grad-True
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # take the current states and predict actions
        #states = states.view(1, BATCH_SIZE, self.num_agents, -1)
        actions_pred = agent.actor_local(states)
        #print (actions_pred.shape)
        #actions_pred = torch.cat(actions_pred, dim=1)
        # -1 * (maximize) Q value for the current prediction
        actor_loss = -agent.critic_local(states.view(BATCH_SIZE,-1), actions_pred.view(BATCH_SIZE,-1)).mean()
        # Minimize the loss
        # zero_grad because we do not want to accumulate 
        # gradients from other batches, so needs to be cleared
        agent.actor_optimizer.zero_grad()
        # compute derivatives for all variables that
        # requires_grad-True
        actor_loss.backward()
        # update those variables that requires_grad-True
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(agent.critic_local, agent.critic_target, TAU)
        self.soft_update(agent.actor_local, agent.actor_target, TAU) 
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


g_agents = MADDPG(in_actor=g_state_size, in_critic=(g_state_size*num_g_agents), action_size=g_action_size, num_agents=num_g_agents, random_seed=0)
s_agents = MADDPG(in_actor=s_state_size, in_critic=(s_state_size*num_s_agents), action_size=s_action_size, num_agents=num_s_agents, random_seed=0)

n_episodes = 10000
print_every = 100


from collections import deque

def ddpg(n_episodes=2000, max_t=1000):
    g_scores_deque = deque(maxlen=100)
    s_scores_deque = deque(maxlen=100)
    g_scores_list = []
    s_scores_list = []
    team0_score_deque = deque(maxlen=100)
    team1_score_deque = deque(maxlen=100)
    team0_score_list = []
    team1_score_list = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)                 # reset the environment    
        g_states = env_info[g_brain_name].vector_observations  # get initial state (goalies)
        s_states = env_info[s_brain_name].vector_observations  # get initial state (strikers)
        s_agents.reset()
        g_agents.reset()
        g_scores = np.zeros(num_g_agents)                      # initialize the score (goalies)
        s_scores = np.zeros(num_s_agents)                      # initialize the score (strikers)
        
        for t in range(max_t):
            g_actions = g_agents.act(g_states)
            s_actions = s_agents.act(s_states)
            g_actions_index = np.argmax(g_actions, axis=1)
            s_actions_index = np.argmax(s_actions, axis=1)
            actions = dict(zip([g_brain_name, s_brain_name], 
                           [g_actions_index, s_actions_index]))
            env_info = env.step(actions)
            # get next states
            g_next_states = env_info[g_brain_name].vector_observations         
            s_next_states = env_info[s_brain_name].vector_observations
            
            # get reward and update scores
            g_rewards = env_info[g_brain_name].rewards  
            s_rewards = env_info[s_brain_name].rewards
            g_scores += g_rewards
            s_scores += s_rewards
            
            # check if episode finished
            g_dones = env_info[g_brain_name].local_done
            s_dones = env_info[s_brain_name].local_done

            g_agents.step(g_states, g_actions, g_rewards, g_next_states, g_dones)
            s_agents.step(s_states, s_actions, s_rewards, s_next_states, s_dones)
            
            g_state = g_next_states
            s_state = s_next_states
            
            g_scores += g_rewards
            s_scores += s_rewards
            if np.any(g_dones) or np.any(s_dones):
                print('\tSteps: ', t)
                break 
        # g_scores = g_scores[np.argmax(g_scores)]
        g_scores_deque.append(np.mean(g_scores))
        g_scores_list.append(np.mean(g_scores))
        # s_scores = s_scores[np.argmax(s_scores)]
        s_scores_deque.append(np.mean(s_scores))
        s_scores_list.append(np.mean(s_scores))

        team0_score = g_scores[0] + s_scores[0]
        team1_score = g_scores[1] + s_scores[1]

        team0_score_list.append(team0_score)
        team1_score_list.append(team1_score)
        
        team0_score_deque.append(team0_score)
        team1_score_deque.append(team1_score)

        print('\rEpisode {}\tAverage Team0 Score: {:.2f}\t T0Score: {:.3f}\t TG: {:.3f}\t TS: {:.3f}'.format(i_episode,
                                                                          np.mean(team0_score_deque), team0_score,
                                                                          g_scores[0], 
                                                                         s_scores[0]))
        print('\rEpisode {}\tAverage Team1 Score: {:.2f}\t T1Score: {:.3f}\t TG: {:.3f}\t TS: {:.3f}'.format(i_episode,
                                                                          np.mean(team1_score_deque), team1_score,
                                                                          g_scores[1], 
                                                                         s_scores[1]))
        print('\rEpisode {}\tAverage GScore: {:.2f}\tGScore: {:.3f}'.format(i_episode, 
                                                                          np.mean(g_scores_deque), 
                                                                         np.max(g_scores)))
        print('\rEpisode {}\tAverage SScore: {:.2f}\tSScore: {:.3f}'.format(i_episode, 
                                                                          np.mean(s_scores_deque), 
                                                                         np.max(s_scores)))
        g_average_score = np.mean(g_scores_deque)
        s_average_score = np.mean(s_scores_deque)
        average_score = g_average_score + s_average_score
        if i_episode % print_every == 10 or g_average_score > 1.0 or s_average_score > 1.0 :
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
            torch.save(g_agents.agents[0].actor_local.state_dict(), 'soccer_g_checkpoint_actor_0.pth')
            torch.save(g_agents.agents[0].critic_local.state_dict(), 'soccer_g_checkpoint_critic_0.pth') 
            torch.save(g_agents.agents[1].actor_local.state_dict(), 'soccer_g_checkpoint_actor_1.pth')
            torch.save(g_agents.agents[1].critic_local.state_dict(), 'soccer_g_checkpoint_critic_1.pth') 
            torch.save(s_agents.agents[0].actor_local.state_dict(), 'soccer_s_checkpoint_actor_0.pth')
            torch.save(s_agents.agents[0].critic_local.state_dict(), 'soccer_s_checkpoint_critic_0.pth') 
            torch.save(s_agents.agents[1].actor_local.state_dict(), 'soccer_s_checkpoint_actor_1.pth')
            torch.save(s_agents.agents[1].critic_local.state_dict(), 'soccer_s_checkpoint_critic_1.pth') 
            if g_average_score > 1.0 or s_average_score > 1.0:
                break
    return g_scores_list, s_scores_list, team0_score_list, team1_score_list

g_scores, s_scores,team0_score_list, team1_score_list = ddpg(n_episodes=n_episodes)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(np.arange(1, len(g_scores)+1), g_scores)
ax = fig.add_subplot(212)
plt.plot(np.arange(1, len(s_scores)+1), s_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


g_agents.agents[0].actor_local.load_state_dict(torch.load('soccer_g_checkpoint_actor_0.pth'))
g_agents.agents[0].critic_local.load_state_dict(torch.load('soccer_g_checkpoint_critic_0.pth'))
g_agents.agents[1].actor_local.load_state_dict(torch.load('soccer_g_checkpoint_actor_1.pth'))
g_agents.agents[1].critic_local.load_state_dict(torch.load('soccer_g_checkpoint_critic_1.pth'))
s_agents.agents[0].actor_local.load_state_dict(torch.load('soccer_s_checkpoint_actor_0.pth'))
s_agents.agents[0].critic_local.load_state_dict(torch.load('soccer_s_checkpoint_critic_0.pth'))
s_agents.agents[1].actor_local.load_state_dict(torch.load('soccer_s_checkpoint_actor_1.pth'))
s_agents.agents[1].critic_local.load_state_dict(torch.load('soccer_s_checkpoint_critic_1.pth'))

env_info = env.reset(train_mode=False)                 # reset the environment    
g_states = env_info[g_brain_name].vector_observations  # get initial state (goalies)
s_states = env_info[s_brain_name].vector_observations  # get initial state (strikers)
g_scores = np.zeros(num_g_agents)                      # initialize the score (goalies)
s_scores = np.zeros(num_s_agents)                      # initialize the score (strikers)
while True:
    # select actions and send to environment
    g_actions = np.random.randint(g_action_size, size=num_g_agents)
    s_actions = np.random.randint(s_action_size, size=num_s_agents)
    actions = dict(zip([g_brain_name, s_brain_name], 
                       [g_actions, s_actions]))
    env_info = env.step(actions)                       

    # get next states
    g_next_states = env_info[g_brain_name].vector_observations         
    s_next_states = env_info[s_brain_name].vector_observations

    # get reward and update scores
    g_rewards = env_info[g_brain_name].rewards  
    s_rewards = env_info[s_brain_name].rewards
    g_scores += g_rewards
    s_scores += s_rewards

    # check if episode finished
    done = np.any(env_info[g_brain_name].local_done)  

    # roll over states to next time step
    g_states = g_next_states
    s_states = s_next_states                            
    if np.any(done):
        print ("Final score {}".format(g_scores+s_scores))
        break

env.close()
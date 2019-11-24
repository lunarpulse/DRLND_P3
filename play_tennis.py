from unityagents import UnityEnvironment
import numpy as np

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
        self.memory.add(state.reshape((1, self.num_agents, -1)), action.reshape((1, self.num_agents, -1)), \
                        reward.reshape((1, self.num_agents, -1)), next_state.reshape((1,self.num_agents, -1)), \
                        done.reshape((1, self.num_agents, -1)))
        
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

agents = MADDPG(in_actor=state_size, in_critic=state_size*num_agents, action_size=action_size, num_agents=num_agents, random_seed=0)

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

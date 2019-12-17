import torch.optim as optim

from model import Actor, Critic, ReplayBuffer, OUNoise, np, torch, F
from prioritized_memory import Memory

BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 5e-4        # learning rate of the critic
WEIGHT_DECAY_ACTOR = 0.0        # L2 weight decay of ACTOR
WEIGHT_DECAY_CRITIC = 0.0        # L2 weight decayof CRITIC
UPDATE_EVERY = 4       # how often to update the network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG():
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """init the agent"""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed
        self.num_agents = num_agents
        
        # Construct Actor networks
        self.actor_local = Actor(state_size, action_size, self.seed).to(device)
        self.actor_target = Actor(state_size, action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY_ACTOR)
        
        # Construct Critic networks 
        self.critic_local = Critic(num_agents * state_size, num_agents * action_size, self.seed).to(device)
        self.critic_target = Critic(num_agents * state_size, num_agents * action_size, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY_CRITIC)
        
        # noise processing
        self.noise = OUNoise((action_size), random_seed)
        
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        # convert state from numpy to pytorch array 
        state = torch.from_numpy(state).float().to(device)
        # use actor_local to predict action
        # turn nn into evaluation mode 
        self.actor_local.eval()
        # turn off computing gradients
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        # turn the nn into training mode
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        
        # clipping the action from min to max
        return np.clip(action, -1, 1)
    
    def target_act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        # convert state from numpy to pytorch array 
        # state = torch.from_numpy(state).float().to(device)
        action = self.actor_target(state)
        return action

    def reset(self):
        """ reset noise """
        self.noise.reset()

class MADDPG:
    def __init__(self, state_size, action_size, num_agents, random_seed):
        self.agents = [DDPG(state_size, action_size, num_agents, random_seed), 
                      DDPG(state_size, action_size, num_agents, random_seed)]
        self.memory = Memory(BUFFER_SIZE)
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.t_step = 0
        
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
        # Compute error
        for ai in range(self.num_agents):
            agent = self.agents[ai]
            action_current = agent.act(state, add_noise = False)
            state_input = torch.from_numpy(state).float().to(device).view(1, -1)
            action_current_tensor = torch.from_numpy(action_current).float().to(device)
            old_val = agent.critic_local.forward(state_input, action_current_tensor.view(1, -1)).detach().cpu().numpy()
            if done[ai]:
                new_val = reward[ai]
            else:
                new_val = reward[ai] + GAMMA * agent.critic_target(state_input, action_current_tensor.view(1, -1)).detach().cpu().numpy()

            # Get priorities and update
            error = abs(old_val - new_val)

            # Save experience / reward
            state = np.asanyarray(state)
            action = np.asanyarray(action)
            reward = np.asanyarray(reward)
            next_state = np.asanyarray(next_state)
            done = np.asanyarray(done)
            self.memory.add(error,
                            (state.reshape((1, self.num_agents, -1)), action.reshape((1, self.num_agents, -1)), \
                            reward.reshape((1, self.num_agents, -1)), next_state.reshape((1,self.num_agents, -1)), \
                            done.reshape((1, self.num_agents, -1))
                            ))
            
        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BUFFER_SIZE / 3:
                for ai in range(self.num_agents):
                    mini_batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)
                    self.learn(mini_batch, idxs, is_weights, ai, GAMMA)
    
    def reset(self):
        [agent.reset() for agent in self.agents]
        
    def learn(self, experience_batch, idxs, is_weights, ai, gamma):
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
        
        mini_batch = [*zip(*experience_batch)] #https://stackoverflow.com/questions/4937491/matrix-transpose-in-python

        states = torch.FloatTensor(np.vstack(mini_batch[0])).to(device)
        actions = torch.FloatTensor(list(mini_batch[1])).to(device)
        rewards = (np.array([*zip(*mini_batch[2])]).transpose())[:,ai].reshape(BATCH_SIZE, -1)
        next_states = torch.FloatTensor(np.vstack(mini_batch[3])).to(device)
        dones = (np.array([*zip(*mini_batch[4])]).transpose())[:,ai].reshape(BATCH_SIZE, -1)

        # bool to binary
        dones = dones.astype(int)

        agent = self.agents[ai]
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        next_states = next_states.view(1, BATCH_SIZE, self.num_agents, -1)
        actions_next = self.target_act(next_states)
        actions_next = torch.cat(actions_next, dim=1)

        next_states = next_states.view(BATCH_SIZE,-1)
        actions_next = actions_next.view(BATCH_SIZE,-1)

        Q_targets_next = agent.critic_target(next_states, actions_next)
        Q_targets_next_np = Q_targets_next.detach().cpu().numpy()
        # Compute Q targets for current states (y_i)

        dones_flipped = np.array([1 - x for x in dones])
        # Q_targets = rewards[:,ai] + (gamma * Q_targets_next * (1 - dones[:,ai]))
        Q_targets_np = rewards + (gamma * np.multiply(Q_targets_next_np, dones_flipped))
        Q_targets = torch.FloatTensor(Q_targets_np).to(device)
        
        # Compute critic loss
        Q_expected = agent.critic_local(states.view(BATCH_SIZE,-1), actions.view(BATCH_SIZE,-1))
        Q_expected_np = Q_expected.detach().cpu().numpy()
        # Get priorities and update
        errors = abs(Q_expected_np - Q_targets_np)
        # update priority
        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        # Minimize the loss
        # compute weighted loss
        # mean squared error loss
        critic_loss = (torch.FloatTensor(is_weights).to(device) * F.mse_loss(Q_expected, Q_targets)).mean()
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
        actions_pred = agent.actor_local(states)
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



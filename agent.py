import torch.optim as optim

from model import Actor, Critic, ReplayBuffer, OUNoise, np, torch, F

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG_agent():
    def __init__(self, in_actor, in_critic, action_size, num_agents, random_seed):
        """init the agent"""
        #self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed
        
        # Construct Actor networks
        self.actor_local = Actor(in_actor, self.action_size, self.seed).to(device)
        self.actor_target = Actor(in_actor, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        #print ("AGENTS {}".format(num_agents))
        # Construct Critic networks 
        self.critic_local = Critic(in_critic, num_agents*self.action_size, self.seed).to(device)
        self.critic_target = Critic(in_critic, num_agents*self.action_size, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
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
        #state = torch.from_numpy(state).float().to(device)
        action = self.actor_target(state)
        return action
        
    def reset(self):
        """ reset noise """
        self.noise.reset()

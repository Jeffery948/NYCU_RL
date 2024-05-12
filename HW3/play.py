import gym
import numpy as np
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network

        self.layer1 = nn.Linear(num_inputs, hidden_size)
        nn.init.kaiming_normal_(self.layer1.weight)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        nn.init.kaiming_normal_(self.layer2.weight)
        self.layer3 = nn.Linear(hidden_size, num_outputs)
        nn.init.kaiming_normal_(self.layer3.weight)
        
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        
        x = F.relu(self.layer1(inputs))
        x = F.relu(self.layer2(x))
        x = F.tanh(self.layer3(x))
        return x
        
        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own critic network

        self.layer1 = nn.Linear(num_inputs, hidden_size)
        nn.init.kaiming_normal_(self.layer1.weight)
        self.layer2 = nn.Linear(hidden_size + num_outputs, hidden_size)
        nn.init.kaiming_normal_(self.layer2.weight)
        self.layer3 = nn.Linear(hidden_size, 1)
        nn.init.kaiming_normal_(self.layer3.weight)

        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network
        
        x = F.relu(self.layer1(inputs))
        x = F.relu(self.layer2(torch.cat([x, actions], dim = 1)))
        x = self.layer3(x)
        return x
        
        ########## END OF YOUR CODE ##########        
        

class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.99, tau=0.002, hidden_size=128, lr_a=1e-4, lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state)))
        mu = mu.data

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed 

        noise = None
        if action_noise == None:
            noise = [0]
        else:
            noise = action_noise.noise()
        noise = torch.FloatTensor(noise)
        return torch.clamp(mu + noise, -1.0, 1.0)

        ########## END OF YOUR CODE ##########
    
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))
    
def test(name1, name2, n_episodes=30):
    """
        Test the learned model (no change needed)
    """     
    model = DDPG(env.observation_space.shape[0], env.action_space)
    model.load_model('./preTrained/{}'.format(name1), './preTrained/{}'.format(name2))
    
    max_episode_len = 10000
    r = []
    
    for i_episode in range(1, n_episodes+1):
        state = torch.Tensor([env.reset()])
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            next_state, reward, done, _ = env.step(action.numpy()[0])
            next_state = torch.Tensor([next_state])
            running_reward += reward
            #env.render()
            if done:
                break
            state = next_state
        r.append(running_reward)
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    r = np.array(r)
    print("Average reward: {}".format(r.mean()))
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    env = gym.make('HalfCheetah')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)
    test('ddpg_actor_HalfCheetah_05102024_050521_.pth', 'ddpg_critic_HalfCheetah_05102024_050521_.pth')
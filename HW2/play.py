import gym
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()
        
        ########## YOUR CODE HERE (5~10 lines) ##########

        self.shared_layer1 = nn.Linear(self.observation_dim, self.hidden_size)
        nn.init.kaiming_normal_(self.shared_layer1.weight)
        self.shared_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.kaiming_normal_(self.shared_layer2.weight)
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)
        nn.init.kaiming_normal_(self.action_layer.weight)
        self.value_layer = nn.Linear(self.hidden_size, 1)
        nn.init.kaiming_normal_(self.value_layer.weight)

        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########

        x = F.relu(self.shared_layer1(state))
        x = F.relu(self.shared_layer2(x))
        action_prob = F.softmax(self.action_layer(x), dim = -1)
        #action_prob = self.action_layer(x)
        state_value = self.value_layer(x)

        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########

        state = torch.tensor(state)
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()

        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########

        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = returns.float()
        returns = (returns - returns.mean()) / (returns.std())
        for (log_prob, state_value), r in zip(saved_actions, returns):
            policy_loss = -log_prob * (r - state_value.detach())
            policy_losses.append(policy_loss)
            value_loss = F.mse_loss(state_value, r)
            value_losses.append(value_loss)
        
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 10000
    r = []
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        r.append(running_reward)
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    r = np.array(r)
    print("Average reward: {}".format(r.mean()))
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 0.01
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)
    test(f'LunarLander0.87_{lr}.pth')
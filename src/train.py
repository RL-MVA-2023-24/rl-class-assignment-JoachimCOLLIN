
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import random
import matplotlib.pyplot as plt

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
class ProjectAgent:
    
    def __init__(self, config=None, model=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config == None:
            config = {
                'learning_rate': 0.0005,
                'gamma': 0.95,
                'buffer_size': 1000000,
                'epsilon_min': 0.03,
                'epsilon_max': 1.,
                'epsilon_decay_period': 7000,
                'epsilon_delay_decay': 100,
                'batch_size': 800,
                'gradient_steps': 4,
                'update_target_strategy': 'replace',
                'update_target_freq': 70,
                'update_target_tau': 0.005,
                'criterion': torch.nn.SmoothL1Loss(),
                'nb_neurons':128
                }

        self.nb_neurons=config['nb_neurons']
        self.nb_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        
        if model == None:
            model = torch.nn.Sequential(
                nn.Linear(self.state_dim, self.nb_neurons),
                nn.ReLU(),
                nn.Linear(self.nb_neurons, self.nb_neurons),
                nn.ReLU(), 
                nn.Linear(self.nb_neurons, self.nb_neurons),
                nn.ReLU(), 
                nn.Linear(self.nb_neurons, self.nb_neurons),
                nn.ReLU(), 
                nn.Linear(self.nb_neurons, self.nb_neurons),
                nn.ReLU(), 
                nn.Linear(self.nb_neurons, self.nb_neurons),
                nn.ReLU(), 
                nn.Linear(self.nb_neurons, self.nb_actions)
                ).to(device)

        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

        # saving best model weights
        self.path_model = 'models/{model_name}.pt'
        self.best_model = None
        self.best_value = 0
        

        
    @staticmethod
    def greedy_action(network, state):
        """
        take a gready action with respect to Q
        """
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()


    def save(self, model, model_name='base'):
        """
        save model in folder /models
        """
        print(model_name)
        if len(model_name.split('_'))>1:
            model_score = int(model_name.split('_')[-1])
            model_score = f'{model_score:,}'
            model_name = '_'.join(model_name.split('_')[0:-1])

        else:
            model_score = '?'
        print(f'Saving model {model_name} with a score of {model_score}...' )
        torch.save(model.state_dict(), self.path_model.format(model_name=model_name + '_' + model_score))
        print('Saving finished ... ')


    def load(self, model_name='layer7_neurons128_28,671,221,016'):
        """
        load model from folder /models
        """
        if len(model_name.split('_'))>1:
            model_score = model_name.split('_')[-1]
            model_name = '_'.join(model_name.split('_')[0:-1])

        else:
            model_score = '?'
        print(f'Loading weights from {model_name} model woth score {model_score}...')
        self.model.load_state_dict(torch.load(self.path_model.format(model_name=model_name + ('_' +model_score)*(model_score != '?'))))
        print('Loading finished ...')

    def act(self, state, use_random=False, epsilon=1.0):
        """
        if use_random = True, act greedily with probabilty 1-epsilon, else act random
        if use_random = False, act only greedily
        """
        if use_random:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
        else:
            action = self.greedy_action(self.model, state)
        
        return action
    

    def gradient_step(self):
        """
        update the model parameters
        """
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode, model_name = None):
        """
        train the DQN model
        """
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            action = self.act(state, use_random=True, epsilon=epsilon)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()

                if (len(episode_return)== 0) or (len(episode_return)>0 and episode_cum_reward > np.max(episode_return)):
                    self.best_model = deepcopy(self.model) 
                    self.best_value = int(episode_cum_reward)

                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
                

            else:
                state = next_state

        # saving model
        if model_name == None:
            self.save(self.best_model, 'base')
        else:
            self.save(self.best_model, model_name + f'_{self.best_value}')

        return episode_return



# Train agent
# agent = ProjectAgent()
# scores = agent.train(env, 200, model_name='layer7_neurons128')
# plt.plot(scores)
# plt.show()
   

    

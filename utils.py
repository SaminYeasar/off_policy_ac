import numpy as np
import random
import os
import time
import json
import torch
# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self):
        self.storage = []

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage.append(data)
    def sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d= [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))


        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


create_folder = lambda f: [ os.makedirs(f) if not os.path.exists(f) else False ]









class Logger(object):
      def __init__(self, args, experiment_name='', environment_name='', argument='', folder='./results'):
            """
            Original: Original implementation of the algorithms
            HDR: Used Qhat
            HDR_RG: Uses Qhat where graph is retained
            DR: Uses Qhat-Vhat
            DR_RG: Uses
            """
            self.rewards = []
              
            self.save_folder = '{}_{}'.format(os.path.join(folder, experiment_name, argument, environment_name, time.strftime('%y-%m-%d-%H-%M-%s')), args.seed)

            create_folder(self.save_folder)
            self.returns_critic_loss = []
            self.returns_reward_loss = []
            self.returns_actor_loss = []
            self.returns_Q_theta = []
            self.returns_True_Q = []

      def record_reward(self, reward_return):
            self.returns_eval = reward_return

      def training_record_reward(self, reward_return):
            self.returns_train = reward_return

      def record_critic_loss(self, critic_loss):
          self.returns_critic_loss.append(critic_loss)

      def record_reward_loss(self, reward_loss):
          self.returns_reward_loss.append(reward_loss)

      def record_actor_loss(self, actor_loss):
          self.returns_actor_loss.append(actor_loss)

      def record_Q_theta(self, Q_theta):
          self.returns_Q_theta.append(Q_theta)

      def record_True_Q(self, True_Q):
          self.returns_True_Q.append(True_Q)

      def save(self):
            np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.returns_eval)

      def save_2(self):
            np.save(os.path.join(self.save_folder, "returns_train.npy"), self.returns_train)

      def save_critic_loss(self):
            np.save(os.path.join(self.save_folder, "critic_loss.npy"), self.returns_critic_loss)

      def save_reward_loss(self):
            np.save(os.path.join(self.save_folder, "reward_loss.npy"), self.returns_reward_loss)

      def save_actor_loss(self):
            np.save(os.path.join(self.save_folder, "actor_loss.npy"), self.returns_actor_loss)

      def save_Q_theta(self):
          np.save(os.path.join(self.save_folder, "Q_theta.npy"), self.returns_Q_theta)

      def save_True_Q(self):
          np.save(os.path.join(self.save_folder, "True_Q.npy"), self.returns_True_Q)


      def save_policy(self, policy):
          torch.save(policy.actor.state_dict(), '%s/actor.pth' % (self.save_folder))
          torch.save(policy.critic.state_dict(), '%s/critic.pth' % (self.save_folder))
            #policy.save(directory=self.save_folder)

      def save_args(self, args):
            """
            Save the command line arguments
            """
            with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
                  json.dump(dict(args._get_kwargs()), f)

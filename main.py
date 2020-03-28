### TODO : Add noise to the rewards - make the MDP noisy

import numpy as np
import torch
import gym
import argparse
import os

import utils
from utils import Logger
from TD3 import TD3
from DDPG import DDPG
from SAC import SAC
from torch.utils.tensorboard import SummaryWriter
from utils import create_folder

def create_policy(args, state_dim, action_dim, max_action):
    if args.policy_name == 'SAC':
        return SAC.SAC(args, state_dim, action_dim, max_action, args.initial_temperature)
    elif args.policy_name == "TD3":
        return TD3.TD3(args, state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG":
        return DDPG.DDPG(args, state_dim, action_dim, max_action)
    assert 'Unknown policy: %s' % args.policy_name


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default='DDPG', help='SAC')	 # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v2")			 # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)					 # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)		 # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)			 # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)		 # Max time steps to run environment for
    parser.add_argument("--save_models", default=True, type=bool)			         # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)		 # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=200, type=int)			 # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)			 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)				 # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)		 # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)		 # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)			 # Frequency of delayed policy updates
    parser.add_argument("--ent_weight", default=0.01, type=float)		 # Range to clip target policy noise
    parser.add_argument("--folder", type=str, default='./results/')

    parser.add_argument("--use_logger", default=True, type=bool, help='whether to use logging or not')
    parser.add_argument("--initial_temperature", default=0.2, type=float)  # SAC temperature
    parser.add_argument("--learn_temperature", type=bool, default=False)  # Whether or not learn the temperature

    parser.add_argument("--use_noise_rewards", action="store_true", default=False, help='whether to use noisy rewards or not')
    parser.add_argument("--reward_noise", default=0.5, type=float)
    parser.add_argument("--lmbda", default=0.8, type=float)





    args = parser.parse_args()





    if args.use_logger:
        file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))

        logger = Logger(args, experiment_name=args.policy_name, environment_name=args.env_name, argument='lambda_{}'.format(args.lmbda), folder='{}'.format(args.folder))
        logger.save_args(args)

        print('Saving to', logger.save_folder)
        writer = SummaryWriter(log_dir='{}/logs/'.format(logger.save_folder))
    else:
        logger = None


    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(args.env_name)

    # Set seeds
    #seed = np.random.randint(10)
    seed = args.seed
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.use_logger:
        print ("---------------------------------------")
        print ("Settings: %s" % (file_name))
        print ("Seed : %s" % (seed))
        print ("---------------------------------------")


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy:
    policy = create_policy(args, state_dim, action_dim, max_action)
    # Initialize Buffer:
    replay_buffer = utils.ReplayBuffer()


    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)]
    episode_reward = 0
    training_evaluations = [episode_reward]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    while total_timesteps < args.max_timesteps:

        if done:

            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward))

                policy.train(logger, args, env, replay_buffer, episode_timesteps, total_timesteps, writer, lmbda=args.lmbda)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:

                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(policy))

                if args.use_logger:
                    logger.record_reward(evaluations)
                    logger.save()
                    logger.save_critic_loss()  # save the critic loss
                    logger.save_reward_loss()
                    logger.save_actor_loss()
                    logger.save_Q_theta()
                    logger.save_True_Q()
                    if args.save_models: logger.save_policy(policy)


            # Reset environment
            obs = env.reset()
            done = False
            training_evaluations.append(episode_reward)

            if args.use_logger:
                logger.training_record_reward(training_evaluations)
                logger.save_2()

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            if args.policy_name == "TD3" or "DDPG":
                action = policy.select_action(np.array(obs))
            elif args.policy_name == "SAC":
                _, action, _ = policy.sample_action(np.array(obs))

        # Perform action
        new_obs, reward, done, _ = env.step(action)

        if args.use_noise_rewards:
            reward = reward + np.random.normal(loc=0.0, scale=args.reward_noise) 


        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(policy))
    training_evaluations.append(episode_reward)
    
    if args.use_logger:
        logger.record_reward(evaluations)
        logger.training_record_reward(training_evaluations)
        logger.save()
        logger.save_2()
        logger.save_critic_loss()  # save the critic loss
        logger.save_reward_loss()
        logger.save_actor_loss()
        logger.save_Q_theta()
        logger.save_True_Q()

        if args.save_models:
            if args.use_logger:
                logger.save_policy(policy)


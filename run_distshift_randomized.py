import tensorflow as tf
from gym_minigrid.wrappers import *
import time, argparse, gym, datetime, os, psutil
from DQN_NOSET import get_DQN_NOSET_agent
from DQN_CP import get_DQN_CP_agent
from DQN_WM import get_DQN_WM_agent
from utils import *
from runtime import generate_comments, evaluate_agent_env_random

process = psutil.Process(os.getpid())

parser = argparse.ArgumentParser(description='')
# arguments for experiment setting
parser.add_argument('--method', type=str, default='DQN_CP', help='DQN_CP, DQN_WM, DQN_NOSET')
parser.add_argument('--game', type=str, default='MiniGrid-RandDistShift', help='')
# arguments for component switches
parser.add_argument('--ignore_TD', type=int, default=0, help='1 to disable value estimator')
parser.add_argument('--ignore_model', type=int, default=0, help='1 to disable model')
parser.add_argument('--prioritized_replay', type=int, default=1, help='0 to turn off prioritized experience replay')
parser.add_argument('--disable_bottleneck', type=int, default=0, help='1 to disable the bottleneck, turn CP into UP')
parser.add_argument('--FC_width', type=int, default=64, help='width of fully connected hidden layers in the transformers')
parser.add_argument('--FC_depth', type=int, default=2, help='depth of fully connected hidden layers in the transformers, 2 means 1 hidden layer')
parser.add_argument('--layers_model', type=int, default=1, help='number of transformer layers in the dynamics model')
parser.add_argument('--len_feature', type=int, default=24, help='length of the feature part in an object')
parser.add_argument('--step_plan_max', type=int, default=5, help='number of planning steps during each tree search MPC session')
parser.add_argument('--type_pos_ebd', type=str, default='cat-learnable', help='cat for non-learnable positional tails')
parser.add_argument('--size_bottleneck', type=int, default=4, help='')
parser.add_argument('--len_ebd_action', type=int, default=8, help='length of the action embeddings')
parser.add_argument('--signal_predict_action', type=int, default=1, help='regularatory loss for representation learning in the model')
parser.add_argument('--noisy_shift', type=int, default=0, help='1 to enable the shift trick mentioned in the Appendix, to enhance generalization')
parser.add_argument('--type_extractor', type=str, default='minigrid_bow', help='if minigrid, use the environment features')
parser.add_argument('--extractor_learnable', type=int, default=1, help='0 to disable learning the encoder')
parser.add_argument('--len_embed_pos', type=int, default=8, help='')
parser.add_argument('--type_compress', type=str, default='semihard', help='or soft')
# arguments that shouldn't be configured
parser.add_argument('--lr', type=float, default=0.00025, help='learning rate of the optimizer(s)')
parser.add_argument('--freq_train_TD', type=int, default=4, help='update interval of value estimator')
parser.add_argument('--freq_train_model', type=int, default=4, help='update interval of model')
parser.add_argument('--gamma', type=float, default=0.99, help='')
parser.add_argument('--size_buffer', type=int, default=1000000, help='')
parser.add_argument('--size_batch', type=int, default=64, help='batch size for updates')
parser.add_argument('--type_optimizer', type=str, default='Adam', help='or RMSprop')
parser.add_argument('--steps_max', type=int, default=50000000, help='')
parser.add_argument('--steps_stop', type=int, default=2500000, help='maximum training steps')
parser.add_argument('--episodes_max', type=int, default=50000000, help='')
parser.add_argument('--freq_eval', type=int, default=200, help='')
parser.add_argument('--seed', type=str, default='', help='')
parser.add_argument('--layernorm', type=int, default=1, help='crucial for reproducibility, 0 to turn off')
parser.add_argument('--atoms_reward', type=int, default=2, help='number of atoms for the distributional output of the estimated reward')
parser.add_argument('--atoms_value', type=int, default=4, help='number of atoms for the distributional output of the estimated value')
parser.add_argument('--value_min', type=float, default=0.0, help='lower bound for the distributional output of the estimated value')
parser.add_argument('--value_max', type=float, default=1.0, help='upper bound for the distributional output of the estimated value')
parser.add_argument('--reward_min', type=float, default=0.0, help='lower bound for the distributional output of the estimated reward')
parser.add_argument('--reward_max', type=float, default=1.0, help='upper bound for the distributional output of the estimated reward')
parser.add_argument('--clip_reward', type=int, default=1, help='clip the reward to its sign, i.e. -1, 0, +1')
parser.add_argument('--n_head', type=int, default=8, help='number of heads for (multi-head) attention operations')
# arguments for identification
parser.add_argument('--comments', type=str, default='x', help='add something here as an identifier')
# NOSET specific
parser.add_argument('--len_hidden', type=int, default=256, help='length of the vectorized representation for NOSET')
# WM specific
parser.add_argument('--period_warmup', type=int, default=int(1e6), help='the warmup period for WM')

args = parser.parse_args()

def get_new_env(env_name='RandDistShift-v0', width=8, height=8, lava_density_range=[0.2, 0.4], min_num_route=2, transposed=False):
    env = gym.make(env_name, width=width, height=height, lava_density_range=lava_density_range, min_num_route=min_num_route, transposed=transposed)
    return env

config_train = {'width': 8, 'height': 8, 'lava_density_range': [0.3, 0.4], 'min_num_route': 1, 'transposed': False} # diff 0.35
env = get_new_env(**config_train)
if args.clip_reward: args.reward_min, args.reward_max = -1.0, 1.0
args.reward_min, args.reward_max = float(min(env.reward_range)), float(max(env.reward_range))
if args.reward_min >= 0: args.value_min = args.reward_min

comment_additional = ''
args = generate_comments(args, additional=comment_additional)
seed = get_set_seed(args.seed, env)

if args.method == 'DQN_NOSET':
    agent = get_DQN_NOSET_agent(env, args)
elif args.method == 'DQN_WM':
    agent = get_DQN_WM_agent(env, args)
elif args.method == 'DQN_UP' or args.method == 'DQN_CP':
    agent = get_DQN_CP_agent(env, args)
elif args.method == 'DQN_Dyna':
    print('Dyna only runs with multiprocessing')
    raise RuntimeError
else:
    raise NotImplementedError

milestones_evaluation, pointer_milestone = [1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 1.5e7, 2e7, 2.5e7, 3e7, 3.5e7, 4e7, 4.5e7, 5e7], 0
writer = tf.summary.create_file_writer("%s/%s/%s/%d" % (env.spec._env_name, args.method, args.comments, seed))
writer.set_as_default()

episode_elapsed = 0
time_start = time.time()
return_cum, step_episode, time_episode_start, str_info = 0, 0, time.time(), ''

while agent.steps_interact <= args.steps_max and episode_elapsed <= args.episodes_max and agent.steps_interact <= args.steps_stop:
    env = get_new_env(**config_train)
    obs_curr, done = env.reset(), False
    if agent.replay_buffer.get_stored_size() > 0: agent.replay_buffer.on_episode_end()
    while not done and agent.steps_interact <= args.steps_max:
        if args.method == 'random':
            obs_next, reward, done, _ = env.step(env.action_space.sample()) # take a random action
            step_episode += 1
        else:
            action = agent.decide(obs_curr, env=env)
            obs_next, reward, done, _ = env.step(action) # take a computed action
            step_episode += 1
            if step_episode == env.unwrapped.max_steps:
                agent.step(obs_curr, action, reward, obs_next, False)
            else:
                agent.step(obs_curr, action, reward, obs_next, done)
        return_cum += reward
        obs_curr = obs_next
    if done:
        time_episode_end = time.time()
        tf.summary.scalar('Return/train', return_cum, agent.steps_interact)
        tf.summary.scalar('Other/episodes', episode_elapsed, agent.steps_interact)
        if agent.initialized and agent.steps_interact >= args.period_warmup:
            if pointer_milestone < len(milestones_evaluation) and agent.steps_interact - args.period_warmup >= milestones_evaluation[pointer_milestone]:
                evaluate_agent_env_random(get_new_env, agent, seed, num_episodes=20, milestone=True)
                pointer_milestone += 1
            elif episode_elapsed % args.freq_eval == 0:
                evaluate_agent_env_random(get_new_env, agent, seed, num_episodes=1, milestone=False)
                if not args.ignore_model and args.step_plan_max:
                    evaluate_agent_env_random(get_new_env, agent, seed, num_episodes=1, milestone=False, suffix='_modelfree', disable_planning=True)
        if args.method == 'random':
            str_info += 'seed: %g, episode: %d, return: %g, steps: %d, sps_episode: %.2f, sps_overall: %.2f' % (seed, episode_elapsed, return_cum, step_episode, step_episode / (time_episode_end - time_episode_start), agent.steps_interact / (time_episode_end - time_start))
        else:
            epsilon = agent.epsilon.value(max(0, agent.steps_interact - args.period_warmup))
            str_info += 'seed: %g, episode: %d, epsilon: %.2f, return: %g, steps: %d' % (seed, episode_elapsed, epsilon, return_cum, step_episode)
            duration_episode = time_episode_end - time_episode_start
            if duration_episode and agent.steps_interact >= agent.time_learning_starts:
                sps_episode = step_episode / duration_episode
                tf.summary.scalar('Other/sps_episode', sps_episode, agent.steps_interact)
                eta = str(datetime.timedelta(seconds=int((args.steps_stop - agent.steps_interact) / sps_episode)))
                str_info += ', sps_episode: %.2f, eta: %s' % (sps_episode, eta)
            print(str_info)
            tf.summary.text('Text/info_train', str_info, step=agent.steps_interact)
        return_cum, step_episode, time_episode_start, str_info = 0, 0, time.time(), ''
        episode_elapsed += 1
time_end = time.time()
env.close()
time_duration = time_end - time_start
print("total time elapsed: %s" % str(datetime.timedelta(seconds=time_duration)))
"""
THIS IS A SCRIPT FOR TESTING AGENTS ON TRADITIONAL STATIC RL SETTINGS
THIS RUNS EXPERIMENT WITH SINGLE PROCESS
ONLY ONE ENVIRONMENT IS GENERATED FOR EACH CASE AND FIXED DURING THE WHOLE PROCESS
"""

import tensorflow as tf
from gym_minigrid.wrappers import *
import time, argparse, gym, datetime, numpy as np, os, psutil

# from DQN import get_DQN_agent
from DQN_NOSET import get_DQN_NOSET_agent
from DQN_CP import get_DQN_CP_agent
from utils import *
from runtime import FrameStack, generate_comments
process = psutil.Process(os.getpid())

parser = argparse.ArgumentParser(description='')
# arguments for experiment setting
parser.add_argument('--method', type=str, default='DQN_NOSET', help='type of agent')
parser.add_argument('--game', type=str, default='RandDistShift', help='RandDistShift or KeyRandDistShift')
parser.add_argument('--version_game', type=str, default='v1', help='v1 (for turn-OR-forward) or v3 (for turn-AND-forward)')
parser.add_argument('--size_world', type=int, default=8, help='the length of each dimension for gridworlds')
parser.add_argument('--color_distraction', type=int, default=0, help='for random colors added to every observation')
# arguments for component switches
parser.add_argument('--ignore_TD', type=int, default=0, help='turn this on to disable value estimator learning')
parser.add_argument('--ignore_model', type=int, default=0, help='turn this on for modelfree agents')
parser.add_argument('--noise_inject', type=int, default=0, help='inject noise into transition model')
parser.add_argument('--prioritized_replay', type=int, default=1, help='prioritized replay buffer, good stuff!')
parser.add_argument('--disable_bottleneck', type=int, default=0, help='1 to disable bottleneck, making CP agent UP')
parser.add_argument('--QKV_depth', type=int, default=1, help='depth of QKV layers in SA layers')
parser.add_argument('--QKV_width', type=int, default=64, help='width of QKV layers in SA layers, does not matter if QKV_depth == 1')
parser.add_argument('--FC_depth', type=int, default=2, help='depth of MLP in transformer layers')
parser.add_argument('--FC_width', type=int, default=64, help='width of MLP in transformer layers')
parser.add_argument('--layers_model', type=int, default=2, help='#action-augmented transformer layers for dynamics model')
parser.add_argument('--len_feature', type=int, default=24, help='length of the feature part of the object embeddings')
parser.add_argument('--len_hidden', type=int, default=256, help='only for NOSET, length of the vectorized representation')
parser.add_argument('--step_plan_max', type=int, default=5, help='#planning steps')
parser.add_argument('--size_bottleneck', type=int, default=8, help='size of bottleneck set')
parser.add_argument('--len_ebd_action', type=int, default=8, help='length of action embedding')
parser.add_argument('--signal_predict_action', type=int, default=1, help='inverse model regularization')
parser.add_argument('--type_extractor', type=str, default='minigrid_bow', help='encoder architecture')
parser.add_argument('--extractor_learnable', type=int, default=1, help='make encoder learnable')
parser.add_argument('--len_embed_pos', type=int, default=8, help='length of positional embedding')
parser.add_argument('--type_attention', type=str, default='semihard', help='could also be soft')
# arguments that shouldn't be configured
parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
parser.add_argument('--freq_train_TD', type=int, default=4, help='training interval for value estimator, every 4 steps train with 1 batch')
parser.add_argument('--freq_train_model', type=int, default=4, help='training interval for transition model')
parser.add_argument('--gamma', type=float, default=0.99, help='discount')
parser.add_argument('--size_buffer', type=int, default=1000000, help='size of replay buffer')
parser.add_argument('--size_batch', type=int, default=64, help='batch size for training')
parser.add_argument('--type_optimizer', type=str, default='Adam', help='feel free to try RMSprop')
parser.add_argument('--framestack', type=int, default=0, help='could use this for Atari')
parser.add_argument('--steps_max', type=int, default=50000000, help='set to be 50M for DQN to perform normally, since exploration period is a percentage')
parser.add_argument('--steps_stop', type=int, default=2500000, help='#agent-environment interactions before the experiment stops')
parser.add_argument('--episodes_max', type=int, default=50000000, help='a criterion just in case we need it')
parser.add_argument('--freq_eval', type=int, default=500, help='interval of periodic evaluation')
parser.add_argument('--seed', type=str, default='', help='if not set manually, would be random')
parser.add_argument('--layernorm', type=int, default=1, help='layer normalization')
parser.add_argument('--atoms_value', type=int, default=4, help='#atoms for value estimator categorical output')
parser.add_argument('--value_min', type=float, default=0.0, help='lower boundary for value estimator output')
parser.add_argument('--value_max', type=float, default=1.0, help='upper boundary for value estimator output')
parser.add_argument('--transform_value', type=int, default=0, help='reserved for environments with varying value magunitudes')
parser.add_argument('--atoms_reward', type=int, default=2, help='#atoms for reward predictor categorical output')
parser.add_argument('--reward_min', type=float, default=0.0, help='lower boundary for reward predictor output')
parser.add_argument('--reward_max', type=float, default=1.0, help='upper boundary for reward predictor output')
parser.add_argument('--transform_reward', type=int, default=0, help='reserved for environments with varying reward magunitudes')
parser.add_argument('--clip_reward', type=int, default=1, help='clip the reward to sign(reward) as in DQN')
parser.add_argument('--n_head', type=int, default=8, help='#heads in the SA layers of (augmented) transformer layers')
parser.add_argument('--period_warmup', type=int, default=int(1e6), help='unsupervised exploration period for WM baselines')
# arguments for runtime
parser.add_argument('--gpu_buffer', type=int, default=0, help='turn this on to put observations in replay onto GPU')
# arguments for identification
parser.add_argument('--comments', type=str, default='x', help='use x for default. If changed, the run will be marked with the string')
# for performance
parser.add_argument('--visualization', type=int, default=0, help='render bottleneck attention')
parser.add_argument('--env_pipeline', type=int, default=1, help='environment generation pipeline, use if many cores')
parser.add_argument('--performance_only', type=int, default=0, help='disable recording debug info, to accelerate experiments')
args = parser.parse_args()

def get_new_env(args, lava_density_range=[0.3, 0.4], min_num_route=1, transposed=False):
    env = gym.make('RandDistShift-%s' % args.version_game, width=args.size_world, height=args.size_world, lava_density_range=lava_density_range, min_num_route=min_num_route, transposed=transposed, random_color=args.color_distraction)
    if args.framestack: env = FrameStack(env, args.framestack)
    return env

config_train = {'lava_density_range': [0.3, 0.4], 'min_num_route': 1, 'transposed': False}
env = get_new_env(args, **config_train)

if args.clip_reward: args.transform_reward, args.reward_min, args.reward_max = False, -1.0, 1.0
args.reward_min, args.reward_max = float(min(env.reward_range)), float(max(env.reward_range))
if args.reward_min >= 0: args.value_min = args.reward_min

comment_additional = ''
args = generate_comments(args, additional=comment_additional)
seed = get_set_seed(args.seed, env)

if args.method == 'DQN':
    raise NotImplementedError('DQN code depracated')
elif args.method == 'DQN_NOSET':
    agent = get_DQN_NOSET_agent(env, args)
elif args.method == 'DQN_UP' or args.method == 'DQN_CP':
    agent = get_DQN_CP_agent(env, args)

milestones_evaluation, pointer_milestone = [1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 1.5e7, 2e7, 2.5e7, 3e7, 3.5e7, 4e7, 4.5e7, 5e7], 0
writer = tf.summary.create_file_writer("%s-%s/%s/%s/%d" % (env.spec._env_name, args.version_game, args.method, args.comments, seed))
writer.set_as_default()

episode_elapsed = 0
time_start = time.time()
return_cum, step_episode, time_episode_start, str_info = 0, 0, time.time(), ''

while agent.steps_interact <= args.steps_max and episode_elapsed <= args.episodes_max and agent.steps_interact <= args.steps_stop:
    obs_curr, done = env.reset(), False
    if agent.replay_buffer.get_stored_size() > 0: agent.replay_buffer.on_episode_end()
    while not done and agent.steps_interact <= args.steps_max:
        if args.method == 'random':
            obs_next, reward, done, _ = env.step(env.action_space.sample()) # take a random action
            reward = np.sign(reward)
            step_episode += 1
        else:
            action = agent.decide(obs_curr, env=env)
            obs_next, reward, done, _ = env.step(action) # take a computed action
            reward = np.sign(reward)
            step_episode += 1
            if step_episode == env.unwrapped.max_steps:
                agent.step(obs_curr, action, reward, obs_next, False)
            else:
                agent.step(obs_curr, action, reward, obs_next, done)
        return_cum += reward
        obs_curr = obs_next
    if done:
        time_episode_end = time.time()
        tf.summary.scalar('Performance/train', return_cum, agent.steps_interact)
        tf.summary.scalar('Other/episodes', episode_elapsed, agent.steps_interact)
        if episode_elapsed and agent.steps_interact >= agent.time_learning_starts:
            if pointer_milestone < len(milestones_evaluation) and agent.steps_interact >= milestones_evaluation[pointer_milestone]:
                evaluate_agent(env, agent, seed, name_method=args.method + args.comments, num_episodes=5, milestone=True)
                pointer_milestone += 1
                if not args.ignore_model and args.step_plan_max:
                    evaluate_agent(env, agent, seed, name_method=args.method + args.comments, num_episodes=5, milestone=True, suffix='_noplan', disable_planning=True)
            elif episode_elapsed % args.freq_eval == 0:
                evaluate_agent(env, agent, seed, name_method=args.method + args.comments, num_episodes=5, milestone=False)
                if not args.ignore_model and args.step_plan_max:
                    evaluate_agent(env, agent, seed, name_method=args.method + args.comments, num_episodes=5, milestone=False, suffix='_noplan', disable_planning=True)
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
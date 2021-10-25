"""
THIS IS THE MAIN SCRIPT FOR EXPERIMENTS
MP STANDS FOR MULTI PROCESSING
"""

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU') 
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True) 
        assert tf.config.experimental.get_memory_growth(gpu) 
except: # Invalid device or cannot modify virtual devices once initialized. 
    pass

import argparse, multiprocessing
from utils import *
from runtime import generate_comments, get_set_seed
import utils_mp, utils_mp_dyna

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # arguments for experiment setting
    parser.add_argument('--method', type=str, default='DQN_CP', help='type of agent')
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
    parser.add_argument('--layers_model', type=int, default=1, help='#action-augmented transformer layers for dynamics model')
    parser.add_argument('--len_feature', type=int, default=24, help='length of the feature part of the object embeddings')
    parser.add_argument('--step_plan_max', type=int, default=5, help='#planning steps')
    parser.add_argument('--size_bottleneck', type=int, default=8, help='size of bottleneck set')
    parser.add_argument('--len_ebd_action', type=int, default=8, help='length of action embedding')
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
    parser.add_argument('--signal_predict_action', type=int, default=1, help='inverse model regularization')
    parser.add_argument('--QKV_depth', type=int, default=1, help='depth of QKV layers in SA layers')
    parser.add_argument('--QKV_width', type=int, default=64, help='width of QKV layers in SA layers, does not matter if QKV_depth == 1')
    parser.add_argument('--FC_depth', type=int, default=2, help='depth of MLP in transformer layers')
    parser.add_argument('--FC_width', type=int, default=64, help='width of MLP in transformer layers')
    # arguments for identification
    parser.add_argument('--comments', type=str, default='x', help='use x for default. If changed, the run will be marked with the string')
    # for multiworker
    parser.add_argument('--gpu_buffer', type=int, default=0, help='turn this on to put observations in replay onto GPU')
    parser.add_argument('--num_explorers', type=int, default=1, help='')
    parser.add_argument('--gpu_explorer', type=int, default=0, help='')
    parser.add_argument('--gpu_evaluator', type=int, default=0, help='')
    # Dyna specific
    parser.add_argument('--learn_dyna_model', type=int, default=0, help='when used with DQN_Dyna, 1 for Dyna, 0 for Dyna*')
    # NOSET specific
    parser.add_argument('--len_hidden', type=int, default=256, help='only for NOSET, length of the vectorized representation')
    # WM specific
    parser.add_argument('--period_warmup', type=int, default=int(1e6), help='unsupervised exploration period for WM baselines')
    # for performance
    parser.add_argument('--visualization', type=int, default=0, help='render bottleneck attention')
    parser.add_argument('--env_pipeline', type=int, default=1, help='environment generation pipeline, use if many cores')
    parser.add_argument('--performance_only', type=int, default=0, help='disable recording debug info, to accelerate experiments')

    args = parser.parse_args()
    env = utils_mp.get_env_minigrid_train(args)

    if args.clip_reward: args.transform_reward, args.reward_min, args.reward_max = False, -1.0, 1.0
    args.reward_min, args.reward_max = float(min(env.reward_range)), float(max(env.reward_range))
    if args.reward_min >= 0: args.value_min = args.reward_min
    args = generate_comments(args, additional='')
    args.seed = get_set_seed(args.seed, env)

    # MAIN
    multiprocessing.set_start_method('spawn', force=True)
    if args.method in 'DQN_Dyna':
        utils_mp_dyna.run_multiprocess(args, utils_mp.get_env_minigrid_train, utils_mp.get_env_minigrid_test)
    else:
        utils_mp.run_multiprocess(args, utils_mp.get_env_minigrid_train, utils_mp.get_env_minigrid_test)
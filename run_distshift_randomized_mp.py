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
    parser.add_argument('--method', type=str, default='DQN_CP', help='DQN_CP, DQN_WM, DQN_Dyna, DQN_NOSET')
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
    # for multiworker
    parser.add_argument('--num_explorers', type=int, default=2, help='number of explorer processes')
    parser.add_argument('--gpu_explorer', type=int, default=0, help='1 to put the explorer agent clones on the gpu')
    parser.add_argument('--gpu_evaluator', type=int, default=0, help='1 to put the evaluator agent clone on the gpu')
    # Dyna specific
    parser.add_argument('--learn_dyna_model', type=int, default=0, help='1 to learn the dyna model, 0 for perfect dynamics')
    # NOSET specific
    parser.add_argument('--len_hidden', type=int, default=256, help='length of the vectorized representation for NOSET')
    # WM specific
    parser.add_argument('--period_warmup', type=int, default=int(1e6), help='the warmup period for WM')
    
    args = parser.parse_args()
    env = utils_mp.get_env_minigrid_train(args)

    if args.clip_reward: args.reward_min, args.reward_max = -1.0, 1.0
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
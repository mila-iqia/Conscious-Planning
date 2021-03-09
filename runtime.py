import tensorflow as tf
import gym, numpy as np, random, sys, os
from gym.envs.registration import register as gym_register

sys.path.append(os.getcwd())
gym_register(id="RandDistShift-v0", entry_point="AdvancedRandDistShift:AdvancedRandDistShift", reward_threshold=0.95)

gpus = tf.config.list_physical_devices('GPU') 
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True) 
        assert tf.config.experimental.get_memory_growth(gpu) 
except: # Invalid device or cannot modify virtual devices once initialized. 
    pass

dict_initializers_pytorch = {'kernel_initializer': 'glorot_normal', 'bias_initializer': tf.keras.initializers.VarianceScaling(scale=1/3.0, mode='fan_in', distribution='uniform')}

return_highest = None
def evaluate_agent(env, agent, seed, num_episodes=10, name_method="default", type_env='minigrid', milestone=False, suffix='', disable_planning=False, step_record=None): #TODO: changed to TF2 but not checked!
    if step_record is None: step_record = agent.steps_interact
    global return_highest
    return_episode, returns = 0, []
    episode = 0
    while episode < num_episodes:
        obs_curr, done = env.reset(), False
        while not done:
            action = agent.decide(obs_curr, eval=True, disable_planning=disable_planning, env=env if type_env == 'minigrid' else None, suffix_record=suffix)
            obs_next, reward, done, _ = env.step(action) # take a computed action
            return_episode += reward
            obs_curr = obs_next
        returns.append(np.copy(return_episode))
        return_episode = 0 # only reset when really done
        episode += 1 # only reset when really done
    return_eval_avg, return_eval_std = np.mean(returns), np.std(returns)
    str_info_eval = 'EVALx%d @ step %d - return_eval_avg: %.2f, return_eval_std: %.2f' % (num_episodes, step_record, return_eval_avg, return_eval_std)
    print(str_info_eval)
    if milestone:
        agent.record_scalar('Return/milestone' + suffix, return_eval_avg, step_record)
    else:
        agent.record_scalar('Return/eval' + suffix, return_eval_avg, step_record)
    tf.summary.text('Text/info_eval' + suffix, str_info_eval, step=step_record)
    if return_highest is None or return_highest < return_eval_avg:
        return_highest = return_eval_avg

def evaluate_agent_env_random(new_env_func, agent, seed, num_episodes=10, milestone=False, suffix='', disable_planning=False, step_record=None): #TODO: changed to TF2 but not checked!
    if step_record is None: step_record = agent.steps_interact
    episode, return_episode, returns = 0, 0, []
    config_test = {'width': 8, 'height': 8, 'lava_density_range': [0.3, 0.4], 'min_num_route': 1, 'transposed': True}
    while episode < num_episodes:
        env = new_env_func(**config_test)
        obs_curr, done = env.reset(), False
        while not done:
            action = agent.decide(obs_curr, eval=True, disable_planning=disable_planning, env=env)
            obs_next, reward, done, _ = env.step(action) # take a computed action
            return_episode += reward
            obs_curr = obs_next
        returns.append(np.copy(return_episode))
        return_episode = 0 # only reset when really done
        episode += 1 # only reset when really done
    return_eval_avg, return_eval_std = np.mean(returns), np.std(returns)
    str_info_eval = 'EVALx%d @ step %d - return_eval_avg: %.2f, return_eval_std: %.2f' % (num_episodes, step_record, return_eval_avg, return_eval_std)
    print(str_info_eval)
    if milestone:
        agent.record_scalar('Return/milestone' + suffix, return_eval_avg, step_record)
    else:
        agent.record_scalar('Return/eval' + suffix, return_eval_avg, step_record)
    tf.summary.text('Text/info_eval' + suffix, str_info_eval, step=step_record)

def filter_nickname(name_env, ram_input=False, type='NoFrameskip'): # or Deterministic?NoFrameskip
    if ram_input:
        return name_env + '-ram' + '%s-v4' % (type)
    else:
        return name_env + '%s-v4' % (type)

def get_set_seed(seed, env):
    if len(seed):
        seed = int(seed)
    else:
        seed = random.randint(0, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        env.seed(seed)
    except:
        print('failed to set env seed')
    return seed

def obs2tensor(obs, divide=None, dtype=tf.float32):
    if isinstance(obs, tf.Tensor):
        tensor = obs
        if len(tensor.shape) == 1 or len(tensor.shape) == 3: tensor = tf.expand_dims(tensor, 0)
    else: # numpy or lazyframes (np.array)
        obs_processed = process_obs(obs)
        tensor = tf.constant(obs_processed)
        del obs_processed
    if divide is not None:
        tensor = tf.math.divide(tensor, divide)
    tensor = tf.dtypes.cast(tensor, dtype)
    return tensor

def process_obs(obs):
    if isinstance(obs, np.ndarray) and (len(obs.shape) == 1 or len(obs.shape) == 3):
        obs = np.expand_dims(obs, 0)
    return obs

def generate_comments(args, additional=''):
    if args.comments == 'x': args.comments = ''
    if len(additional): args.comments += additional
    if not tf.test.is_built_with_cuda(): args.comments += '_CPU'
    if not args.method == 'DQN_WM': args.period_warmup = 0
    if args.method == 'DQN_Dyna': args.signal_predict_action = 0
    if not args.method == 'DQN_NOSET' and not args.method == 'DQN_WM' and args.ignore_model or args.disable_bottleneck and args.method == 'DQN_CP': args.method = 'DQN_UP'
    if not args.prioritized_replay: args.comments += '_noprior'
    if args.method == 'DQN_NOSET':
        args.comments += '_noset'
        if args.ignore_model:
            args.comments += '_modelfree'
        else:
            args.comments += '_%dx%d' % (args.layers_model, args.len_hidden)
    elif args.method != 'DQN' and args.ignore_model:
        args.step_plan_max = 0
        args.comments += '_modelfree'
    else:
        args.comments += '_%dx%dx%d' % (args.layers_model, args.len_feature, args.n_head)
        args.comments += '_FC%dx%d' % (args.FC_depth, args.FC_width)
        if args.ignore_TD:
            args.comments += '_ignore_TD'
        else:
            args.comments += '_%dstep' % args.step_plan_max
        if args.method == 'DQN_CP' or args.method == 'DQN_WM' or args.method == 'DQN_Dyna' and not args.disable_bottleneck: args.comments += '_%dpicks_%s' % (args.size_bottleneck, args.type_compress)
        if args.method == 'DQN_WM': args.comments += '_WM%g' % (args.period_warmup,)
        if args.method == 'DQN_Dyna' and not args.learn_dyna_model: args.comments += '_truemodel'
    if 'minigrid' in args.game.lower():
        if not args.clip_reward: args.comments += '_unclip'
        if 'bow' in args.type_extractor: args.comments += '_bow'
        if 'linear' in args.type_extractor: args.comments += '_linear'
        if 'mlp' in args.type_extractor: args.comments += '_mlp'
        if args.size_batch != 64: args.comments += '_bs%d' % (args.size_batch)
        if args.lr != 0.00025: args.comments += '_lr_%gx' % (args.lr / 0.00025)
    else:
        raise NotImplementedError
    if args.gamma != 0.99: args.comments += '_%.2f' % (args.gamma)
    if args.type_optimizer != 'Adam': args.comments += '_%s' % args.type_optimizer
    if not args.layernorm: args.comments += '_nonorm'
    if args.noisy_shift: args.comments += '_shift'
    if not args.extractor_learnable: args.comments += '_frozen_enc'
    if not args.ignore_model:
        if args.signal_predict_action: args.comments += '_pred_act'
    if args.comments[0] == '_': args.comments = args.comments[1:]
    return args

def get_cpprb_env_dict(env):
    def get_space_size(space):
        if isinstance(space, gym.spaces.box.Box):
            return space.shape
        elif isinstance(space, gym.spaces.discrete.Discrete):
            return [1, ]  # space.n
        else:
            raise NotImplementedError("Assuming to use Box or Discrete, not {}".format(type(space)))
    shape_obs = get_space_size(env.observation_space)
    env_dict = {"obs": {"shape": shape_obs}, "act": {}, "rew": {"shape": 1}, "done": {}} # "dtype", np.bool
    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        env_dict["act"]["shape"] = 1
        env_dict["act"]["dtype"] = np.int32
    elif isinstance(env.action_space, gym.spaces.box.Box):
        env_dict["act"]["shape"] = env.action_space.shape
        env_dict["act"]["dtype"] = np.float32
    obs = env.reset()
    if isinstance(obs, np.ndarray):
        env_dict["obs"]["dtype"] = obs.dtype
    return env_dict

def get_cpprb(env, size_buffer, prioritized=False):
    env_dict = get_cpprb_env_dict(env)
    if prioritized:
        from cpprb import PrioritizedReplayBuffer
        return PrioritizedReplayBuffer(size_buffer, env_dict, next_of=("obs"))
    else:
        from cpprb import ReplayBuffer
        return ReplayBuffer(size_buffer, env_dict, next_of=("obs"))
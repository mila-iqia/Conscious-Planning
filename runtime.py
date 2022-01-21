"""
RUNTIME RELATED STUFFS
"""
import tensorflow as tf
import gym, numpy as np, random, sys, os
from collections import deque
from gym import spaces
from gym.envs.registration import register as gym_register

sys.path.append(os.getcwd())
gym_register(id="RandDistShift-v1", entry_point="RandDistShift:RandDistShift", reward_threshold=0.95)
gym_register(id="RandDistShift-v2", entry_point="RandDistShift2:RandDistShift2", reward_threshold=0.95)
gym_register(id="RandDistShift-v3", entry_point="RandDistShift3:RandDistShift3", reward_threshold=0.95)
gym_register(id="KeyRandDistShift-v3", entry_point="KeyRandDistShift:KeyRandDistShift", reward_threshold=0.95)

gpus = tf.config.list_physical_devices('GPU') 
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True) 
        assert tf.config.experimental.get_memory_growth(gpu) 
except: # Invalid device or cannot modify virtual devices once initialized. 
    pass

dict_initializers_pytorch = {'kernel_initializer': 'glorot_normal', 'bias_initializer': tf.keras.initializers.VarianceScaling(scale=1/3.0, mode='fan_in', distribution='uniform')}

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done: break
        max_frame = self._obs_buffer.max(axis=0) # Note that the observation on the done=True frame doesn't matter
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset. No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0: # for Qbert sometimes we stay in lives == 0 condition for a few frames. so it's important to keep lives > 0, so that we only reset once the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to height x width
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, num_colors), dtype=np.uint8)
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if frame.shape[0] != self._height and frame.shape[1] != self._width:
            frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale: frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

class FrameStack(gym.Wrapper):
    def __init__(self, env, k, gpu=False):
        """ Stack k last frames. Returns lazy array, which is memory efficient. """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.gpu = gpu
        self.frames = deque([], maxlen=k)
        if self.gpu: self.frames_gpu = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
            if self.gpu: self.frames_gpu.append(tf.expand_dims(tf.constant(ob), 0))
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        if self.gpu: self.frames_gpu.append(tf.expand_dims(tf.constant(ob), 0))
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        if self.gpu:
            assert len(self.frames_gpu) == self.k
            return LazyFrames(list(self.frames_gpu), gpu=True)
        else:
            return LazyFrames(list(self.frames))

class LazyFrames(object):
    def __init__(self, frames, gpu=False):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
        This object should only be converted to numpy array before being passed to the model."""
        self._frames = frames
        self._out = None
        self.gpu = gpu

    def _force(self):
        if self._out is None:
            if self.gpu:
                return tf.concat(self._frames, axis=-1)
            else:
                return np.concatenate(self._frames, axis=-1)
        else:
            return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

return_highest = None
def evaluate_agent(env, agent, seed, num_episodes=10, name_method="default", type_env='minigrid', render=False, milestone=False, suffix='', disable_planning=False, step_record=None): #TODO: changed to TF2 but not checked!
    if step_record is None: step_record = agent.steps_interact
    global return_highest
    return_episode, returns = 0, []
    episode = 0
    render = False
    while episode < num_episodes:
        obs_curr, done = env.reset(), False
        while not done:
            action = agent.decide(obs_curr, eval=True, disable_planning=disable_planning, env=env if type_env == 'minigrid' else None, suffix_record=suffix)
            obs_next, reward, done, _ = env.step(action) # take a computed action
            return_episode += reward
            obs_curr = obs_next
        if type_env != 'atari' or env.was_real_done:
            returns.append(np.copy(return_episode))
            return_episode = 0 # only reset when really done
            episode += 1 # only reset when really done
    return_eval_avg, return_eval_std = np.mean(returns), np.std(returns)
    str_info_eval = 'EVALx%d @ step %d - return_eval_avg: %.2f, return_eval_std: %.2f' % (num_episodes, step_record, return_eval_avg, return_eval_std)
    print(str_info_eval)
    if milestone:
        agent.record_scalar('Performance/milestone' + suffix, return_eval_avg, step_record)
    else:
        agent.record_scalar('Performance/eval' + suffix, return_eval_avg, step_record)
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
        agent.record_scalar('Performance/milestone' + suffix, return_eval_avg, step_record)
    else:
        agent.record_scalar('Performance/eval' + suffix, return_eval_avg, step_record)
    tf.summary.text('Text/info_eval' + suffix, str_info_eval, step=step_record)

def filter_nickname(name_env, ram_input=False, type='NoFrameskip'): # or Deterministic?NoFrameskip
    if ram_input:
        return name_env + '-ram' + '%s-v4' % (type)
    else:
        return name_env + '%s-v4' % (type)

def make_atari(env_id, max_episode_steps=None, noop=False, max_skip=True):
    env = gym.make(env_id)
    if noop: env = NoopResetEnv(env, noop_max=30)
    if max_skip: env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def get_env(name_env, size=(84, 84), grayscale=True, ram_input=False, episode_life=True, noop=False, gpu=False):
    env = make_atari(name_env, 27000, noop=noop) # 108K frames cap
    if ram_input:
        env = wrap_deepmind_ram(env, episode_life=episode_life)
    else:
        env = wrap_deepmind(env, size=size, grayscale=grayscale, episode_life=episode_life, gpu=gpu)
    return env

def wrap_deepmind_ram(env, episode_life=True, frame_stack=True):
    if episode_life: env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings(): env = FireResetEnv(env)
    if frame_stack: env = FrameStack(env, 4)
    return env

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

def wrap_deepmind(env, size=(84, 84), grayscale=True, episode_life=True, frame_stack=True, gpu=False):
    height, width = size
    if episode_life: env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings(): env = FireResetEnv(env)
    env = WarpFrame(env, width=width, height=height, grayscale=grayscale)
    if frame_stack: env = FrameStack(env, 4, gpu=gpu)
    return env

def obs2tensor(obs, divide=None, dtype=tf.float32):
    if isinstance(obs, LazyFrames): # lazyframes (tf.tensor)
        obs = obs._force()
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
    if isinstance(obs, LazyFrames):
        obs = np.array(obs, copy=False)
        obs = np.expand_dims(obs, 0)
    elif isinstance(obs, np.ndarray) and (len(obs.shape) == 1 or len(obs.shape) == 3):
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
            if args.noise_inject: args.comments += '_NI'
    elif args.method != 'DQN' and args.ignore_model:
        args.step_plan_max = 0
        args.comments += '_modelfree'
    else:
        args.comments += '_%dx%dx%d' % (args.layers_model, args.len_feature, args.n_head)
        if args.noise_inject: args.comments += '_NI'
        args.comments += '_KQV%dx%d' % (args.QKV_depth, args.QKV_width)
        args.comments += '_FC%dx%d' % (args.FC_depth, args.FC_width)
        if args.ignore_TD:
            args.comments += '_ignore_TD'
        else:
            args.comments += '_%dstep' % args.step_plan_max
        if args.method == 'DQN_CP' or args.method == 'DQN_WM' or args.method == 'DQN_Dyna' and not args.disable_bottleneck: args.comments += '_%dpicks_%s' % (args.size_bottleneck, args.type_attention)
        if args.method == 'DQN_WM': args.comments += '_WM%g' % (args.period_warmup,)
        if args.method == 'DQN_Dyna' and not args.learn_dyna_model: args.comments += '_truemodel'
    if 'minigrid' in args.game.lower() or 'distshift' in args.game.lower():
        if 'key' in args.game.lower():
            args.reward_max = 0.5
            args.atoms_reward = 2
            args.clip_reward = 0
        if args.size_world != 8: args.comments += '_worldsize%g' % (args.size_world)
        if not args.clip_reward: args.comments += '_unclip'
        if 'bow' in args.type_extractor: args.comments += '_bow'
        if 'linear' in args.type_extractor: args.comments += '_linear'
        if 'mlp' in args.type_extractor: args.comments += '_mlp'
        if args.size_batch != 64: args.comments += '_bs%d' % (args.size_batch)
        if args.framestack: args.comments += '_stack%d' % (args.framestack)
        if args.lr != 0.00025: args.comments += '_lr_%gx' % (args.lr / 0.00025)
        if args.color_distraction:
            args.type_task = 'DistractedDistShift-%s' % args.version_game
        elif 'key' in args.game.lower():
            args.type_task = 'KeyRandDistShift-%s' % args.version_game
        else:
            args.type_task = 'RandDistShift-%s' % args.version_game
    elif 'atari' in args.game.lower():
        if args.clip_reward: args.comments += '_clip'
        if args.size_batch != 32: args.comments += '_bs%d' % (args.size_batch)
        if not args.framestack: args.comments += '_nostack'
        if args.lr != 0.0000625: args.comments += '_lr_%gx' % (args.lr / 0.0000625)
    elif 'procgen' in args.game.lower():
        if args.clip_reward: args.comments += '_clip'
        if args.size_batch != 512: args.comments += '_bs%d' % (args.size_batch)
        if args.framestack: args.comments += '_stack%d' % (args.framestack)
        if args.lr != 0.00025: args.comments += '_lr_%gx' % (args.lr / 0.00025)
    if args.transform_value: args.comments += '_trv'
    if args.transform_reward: args.comments += '_trr'
    if args.gamma != 0.99: args.comments += '_%.2f' % (args.gamma)
    if args.type_optimizer != 'Adam': args.comments += '_%s' % args.type_optimizer
    if not args.layernorm: args.comments += '_nonorm'
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
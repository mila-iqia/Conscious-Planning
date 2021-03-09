import time, gym, datetime, numpy as np
from DQN_CP import get_DQN_CP_BASE_agent, get_DQN_CP_agent
from DQN_NOSET import get_DQN_NOSET_BASE_agent, get_DQN_NOSET_agent
from DQN_WM import get_DQN_WM_BASE_agent, get_DQN_WM_agent
from DQN_Dyna import get_DQN_Dyna_BASE_agent, get_DQN_Dyna_agent
from utils import *
from runtime import get_cpprb_env_dict
from multiprocessing import Process, Value, Event
from multiprocessing.managers import SyncManager
from cpprb import ReplayBuffer, MPReplayBuffer, MPPrioritizedReplayBuffer
from utils import *
import os, psutil, copy
from tensorboardX import SummaryWriter

try:
    from gym.envs.registration import register as gym_register
    gym_register(id="RandDistShift-v0", entry_point="AdvancedRandDistShift:AdvancedRandDistShift", reward_threshold=0.95)
except:
    pass

def get_space_size(space):
    if isinstance(space, gym.spaces.box.Box):
        return space.shape
    elif isinstance(space, gym.spaces.discrete.Discrete):
        return [1, ]  # space.n
    else:
        raise NotImplementedError("Assuming to use Box or Discrete, not {}".format(type(space)))

def get_default_rb_dict(size, env):
    return {"size": size, "default_dtype": np.float32,
            "env_dict": {
            "obs": {"shape": get_space_size(env.observation_space)},
            "next_obs": {"shape": get_space_size(env.observation_space)},
            "act": {"shape": get_space_size(env.action_space)},
            "rew": {},
            "done": {}}}

def get_env_procgen(args):
    env = gym.make('procgen:procgen-%s-v0' % (args.game.lower(),), use_backgrounds=False, restrict_themes=True)
    if args.method in ['DQN_CP' or 'DQN_UP'] and args.step_plan_max: assert not env.spec.nondeterministic # does not support stochastic envs
    return env

def get_env_minigrid_train(args, width=8, height=8, lava_density_range=[0.3, 0.4], min_num_route=1, transposed=False): # diff 0.35
    config = {'width': width, 'height': height, 'lava_density_range': lava_density_range, 'min_num_route': min_num_route, 'transposed': transposed}
    env = gym.make('RandDistShift-v0', **config)
    return env

def get_env_minigrid_test(args, width=8, height=8, lava_density_range=[0.3, 0.4], min_num_route=1, transposed=True): # diff 0.35
    config = {'width': width, 'height': height, 'lava_density_range': lava_density_range, 'min_num_route': min_num_route, 'transposed': transposed}
    env = gym.make('RandDistShift-v0', **config)
    return env

def get_agent(env, args, writer, global_rb=None):
    if global_rb is not None:
        if args.method in ['DQN_CP', 'DQN_UP']:
            agent = get_DQN_CP_agent(env, args, replay_buffer=global_rb, writer=writer)
        elif args.method == 'DQN_WM':
            agent = get_DQN_WM_agent(env, args, replay_buffer=global_rb, writer=writer)
        elif args.method == 'DQN_NOSET':
            agent = get_DQN_NOSET_agent(env, args, replay_buffer=global_rb, writer=writer)
        elif args.method == 'DQN_Dyna':
            agent = get_DQN_Dyna_agent(env, args, replay_buffer=global_rb, writer=writer)
        else:
            raise NotImplementedError
    else:
        if args.method in ['DQN_CP', 'DQN_UP']:
            agent = get_DQN_CP_BASE_agent(env, args, writer)
        elif args.method == 'DQN_WM':
            agent = get_DQN_WM_BASE_agent(env, args, writer)
        elif args.method == 'DQN_NOSET':
            agent = get_DQN_NOSET_BASE_agent(env, args, writer)
        elif args.method == 'DQN_Dyna':
            agent = get_DQN_Dyna_BASE_agent(env, args, writer=writer)
        else:
            raise NotImplementedError
    return agent

def prepare_experiment(env, args):
    SyncManager.register('SummaryWriter', SummaryWriter)
    manager = SyncManager()
    manager.start()
    kwargs = get_default_rb_dict(args.size_buffer, env)
    kwargs["check_for_update"] = True
    kwargs['env_dict'] = get_cpprb_env_dict(env)
    kwargs['env_dict']['next_obs'] = kwargs['env_dict']['obs'] # no memory compression for MP else huge problems
    if args.prioritized_replay:
        global_rb = MPPrioritizedReplayBuffer(**kwargs)
    else:
        global_rb = MPReplayBuffer(**kwargs)
    kwargs_local = copy.deepcopy(kwargs)
    kwargs_local['size'] = 128
    # queues to share network parameters between a learner and explorers
    n_queue = args.num_explorers + 1 # for evaluation
    queues = [manager.Queue() for _ in range(n_queue)]
    queue_envs_train, queue_envs_eval = manager.Queue(maxsize=32), manager.Queue(maxsize=32)
    # Event object to share training status. if event is set True, all exolorers stop sampling transitions
    event_terminate = Event()
    # Shared memory objects to count number of samples and applied gradients
    steps_interact, episodes_interact = Value('i', 0), Value('i', 0) # dtype and initial values
    signal_explore = Value('b', False)
    glboal_writer = manager.SummaryWriter("%s/%s/%s/%d" % (args.game, args.method, args.comments, args.seed))
    return global_rb, kwargs_local, queues, queue_envs_train, queue_envs_eval, event_terminate, steps_interact, episodes_interact, signal_explore, glboal_writer

def import_tf():
    import tensorflow as tf
    if tf.config.experimental.list_physical_devices('GPU'):
        for cur_device in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(cur_device, enable=True)
    return tf

def evaluate_agent_mp(func_env, agent, num_episodes=10, type_env='minigrid', suffix='', disable_planning=False, step_record=None, queue_envs=None, heuristic='best_first', record_ts=True):
    if step_record is None: step_record = agent.steps_interact
    return_episode, returns = 0, []
    for _ in range(num_episodes):
        if queue_envs is not None:
            try:
                env = queue_envs.get_nowait()
            except:
                env = func_env()
        else:
            env = func_env()
        obs_curr, done, flag_reset = env.reset(), False, False
        steps_episode, return_episode = 0, 0
        while not flag_reset:
            action = agent.decide(obs_curr, eval=True, disable_planning=disable_planning, env=env if type_env == 'minigrid' else None, suffix_record=suffix, heuristic=heuristic, record_ts=record_ts)
            obs_next, reward, done, info = env.step(action) # take a computed action
            steps_episode += 1
            return_episode += reward
            obs_curr = obs_next
            agent.steps_interact += 1
            flag_reset = done
        returns.append(np.copy(return_episode))
    return_eval_avg, return_eval_std = np.mean(returns), np.std(returns)
    print('EVALx%d @ step %d - return_eval_avg: %.2f, return_eval_std: %.2f' % (num_episodes, step_record, return_eval_avg, return_eval_std))
    agent.record_scalar('Return/eval' + suffix, return_eval_avg, step_record)

def generator_env(queue_envs_train, queue_envs_eval, func_env_train, func_env_eval, event_terminate, args):
    while not event_terminate.is_set():
        flag_q_train_full, flag_q_eval_full = queue_envs_train.full(), queue_envs_eval.full()
        if flag_q_train_full and flag_q_eval_full:
            time.sleep(0.0001)
        else:
            if not flag_q_train_full:
                env_train = func_env_train(args)
                queue_envs_train.put_nowait(env_train)
            if not flag_q_eval_full:
                env_eval = func_env_eval(args)
                queue_envs_eval.put_nowait(env_eval)

def explorer(global_rb, kwargs_local, queue, queue_envs_train, steps_interact, episodes_interact, event_terminate, signal_explore, args, func_env, writer):
    if args.gpu_explorer:
        tf = import_tf()
    else:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    local_rb = ReplayBuffer(**kwargs_local)
    env = func_env(args)
    agent = get_agent(env, args, writer)
    agent.initialize(env.reset(), env.action_space.sample())
    size_submit = 32
    if 'minigrid' in args.game.lower():
        type_env = 'minigrid'
    else:
        raise NotImplementedError
    flag_newenvs = 'randdistshift' in args.game.lower()
    while not event_terminate.is_set():
        return_cum, steps_episode = 0, 0 # return_cum, return_cum_clipped, steps_episode = 0, 0, 0
        obs_curr, done, real_done, flag_reset = env.reset(), False, False, False
        if local_rb.get_stored_size() > 0: local_rb.on_episode_end()
        while not flag_reset:
            while not signal_explore.value and not event_terminate.is_set():
                time.sleep(0.0001)
            if not queue.empty() and agent.initialized:
                dict_shared = None
                while not queue.empty():
                    del dict_shared
                    dict_shared = queue.get_nowait()
                agent.weights_copyfrom(dict_shared)
                del dict_shared
            steps_interact_curr, episodes_interact_curr = steps_interact.value, episodes_interact.value
            agent.steps_interact = steps_interact.value
            action = agent.decide(obs_curr, eval=False, env=env if type_env == 'minigrid' else None, record_ts=writer is not None)
            obs_next, reward, done, info = env.step(action) # take a computed action
            steps_episode += 1
            if type_env == 'procgen':
                real_done = done and steps_episode != env.spec.max_episode_steps and reward == 0 and not info['prev_level_complete']
            elif type_env == 'minigrid':
                real_done = done and steps_episode != env.unwrapped.max_steps
            else:
                real_done = done
            agent.step(obs_curr, action, reward, obs_next, real_done, update=False)
            local_rb.add(obs=obs_curr, act=action, rew=reward, done=real_done, next_obs=obs_next)
            return_cum += reward
            obs_curr = obs_next
            flag_reset = real_done or (done and type_env == 'minigrid')
            if local_rb.get_stored_size() >= size_submit:
                if flag_reset: local_rb.on_episode_end()
                size_local_rb = local_rb.get_stored_size()
                samples_local = local_rb.get_all_transitions()
                local_rb.clear()
                if args.prioritized_replay:
                    global_rb.add(**samples_local, priorities=agent.calculate_priorities(samples_local))
                else:
                    global_rb.add(**samples_local)
                with steps_interact.get_lock(): steps_interact.value += size_local_rb
        if writer is not None:
            writer.add_scalar('Return/train', return_cum, steps_interact_curr)
            writer.add_scalar('Other/episodes', episodes_interact_curr, steps_interact_curr)
        with episodes_interact.get_lock(): episodes_interact.value += 1
        if flag_newenvs:
            del env
            try:
                env = queue_envs_train.get_nowait()
            except:
                env = func_env(args)

def learner(global_rb, queues, steps_interact, episodes_interact, event_terminate, signal_explore, args, pid_main, func_env, writer):
    tf = import_tf()
    process_main = psutil.Process(pid_main)
    process_learner = psutil.Process(os.getpid())
    env = func_env(args)
    agent = get_agent(env, args, writer, global_rb=global_rb)
    step_last_sync, episode_last_eval, time_last_disp = 0, 0, time.time()
    print('[LEARNER] loop enter')
    agent.steps_interact = steps_interact.value
    freq_sync = 64
    flag_updated_since_sync = False
    batch_preload = None
    steps_processed_last_disp, episode_last_disp, time_last_disp = 0, 0, time.time()
    while not event_terminate.is_set():
        episodes_interact_curr = episodes_interact.value
        flag_eval = agent.initialized and (episodes_interact_curr - episode_last_eval) >= args.freq_eval
        agent.steps_interact = steps_interact.value
        flag_sync = agent.initialized and (agent.steps_interact - step_last_sync) >= freq_sync and agent.steps_interact >= agent.time_learning_starts
        flag_need_update = agent.need_update()
        if flag_need_update:
            with signal_explore.get_lock(): signal_explore.value = False
            agent.step_update(batch=batch_preload)
            batch_preload = None
            flag_updated_since_sync = True
            if episodes_interact_curr - episode_last_disp > 0:
                mem = process_main.memory_info().rss
                mem_learner = 0
                for process_child in process_main.children(recursive=True):
                    if process_child.pid == process_learner.pid:
                        mem_learner = process_child.memory_info().rss
                    mem += process_child.memory_info().rss
                mem, mem_learner = mem / 1073741824, mem_learner / 1073741824
                time_from_last_disp = time.time() - time_last_disp
                if time_from_last_disp > 0:
                    sps = (agent.steps_processed - steps_processed_last_disp) / time_from_last_disp
                    if sps > 0:
                        eta = str(datetime.timedelta(seconds=int((args.steps_stop - agent.steps_processed) / sps)))
                        writer.add_scalar('Other/sps', sps, agent.steps_interact)
                        try:
                            print('[LEARNER] episode_explored: %d, step_explored: %d, steps_processed: %d, size_buffer: %d, epsilon: %.2f, mem: %.2f(%.2f)GiB, sps: %.2f, eta: %s' % (episodes_interact_curr, steps_interact.value, agent.steps_processed, global_rb.get_stored_size(), agent.epsilon.value(agent.steps_interact), mem, mem_learner, sps, eta))
                        except:
                            pass
                    else:
                        try:
                            print('[LEARNER] episode_explored: %d, step_explored: %d, steps_processed: %d, size_buffer: %d, epsilon: %.2f, mem: %.2f(%.2f)GiB, sps: 0.00, eta: ---' % (episodes_interact_curr, steps_interact.value, agent.steps_processed, global_rb.get_stored_size(), agent.epsilon.value(agent.steps_interact), mem, mem_learner))
                        except:
                            pass
                else:
                    try:
                        print('[LEARNER] episode_explored: %d, step_explored: %d, steps_processed: %d, size_buffer: %d, epsilon: %.2f, mem: %.2fGiB, sps: inft, eta: 0s' % (episodes_interact_curr, steps_interact.value, agent.steps_processed, global_rb.get_stored_size(), agent.epsilon.value(agent.steps_interact), mem, mem_learner))
                    except:
                        pass
                if np.random.rand() < 0.01: writer.add_scalar('Other/RAM', mem, agent.steps_processed)
                steps_processed_last_disp, episode_last_disp, time_last_disp = agent.steps_processed, episodes_interact_curr, time.time()
            dict_shared = None
        elif batch_preload is None and agent.initialized and global_rb.get_stored_size() >= agent.size_batch:
            batch_preload = agent.sample_batch()
        if (flag_sync and not flag_need_update and flag_updated_since_sync) or flag_eval:
            if args.method != 'DQN_Dyna' and agent.ignore_model:
                dict_shared = {'network_policy_src': agent.network_policy.get_weights(), 'embed_pos_src': tf.keras.backend.get_value(agent.embed_pos), 'model_src': None, 'steps_processed': agent.steps_processed}
            else:
                dict_shared = {'network_policy_src': agent.network_policy.get_weights(), 'embed_pos_src': tf.keras.backend.get_value(agent.embed_pos), 'model_src': agent.model.get_weights(), 'steps_processed': agent.steps_processed}
        if flag_sync and not flag_need_update and flag_updated_since_sync:
            # print('[LEARNER] parameters broadcast to explorers')
            for i in range(len(queues) - 1):
                try:
                    queues[i].put_nowait(dict_shared) # put it in every explorer except the evaluator
                except:
                    print('queue.put_nowait exception')
            step_last_sync += freq_sync
            flag_updated_since_sync = False
        if flag_eval:
            # print('[LEARNER] parameters broadcast to evaluator')
            try:
                queues[-1].put_nowait(dict_shared) # put it in every explorer except the evaluator
            except:
                print('queue.put_nowait exception')
            episode_last_eval += args.freq_eval
        if agent.steps_processed >= min(args.steps_stop, args.steps_max) or episodes_interact_curr >= args.episodes_max:
            event_terminate.set()
        if not flag_need_update:
            with signal_explore.get_lock(): signal_explore.value = True
            writer.flush()

def evaluator(steps_interact, event_terminate, queue, queue_envs_eval, args, func_env, writer):
    if args.gpu_evaluator:
        tf = import_tf()
    else:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    env = func_env(args)
    agent = get_agent(env, args, writer)
    agent.initialize(env.reset(), env.action_space.sample())
    if 'minigrid' in args.type_extractor.lower():
        type_env = 'minigrid'
    else:
        raise NotImplementedError
    print('[EVALUATOR] loop enter')
    flag_newenvs = 'randdistshift' in args.game.lower()
    if not flag_newenvs: queue_envs_eval = None
    while not event_terminate.is_set():
        if queue.empty():
            time.sleep(0.0001)
        else:
            # print('[EVALUATOR] parameters clone, eval call')
            dict_shared = None
            while not queue.empty():
                del dict_shared
                dict_shared = queue.get_nowait()
            agent.weights_copyfrom(dict_shared)
            steps_interact = dict_shared['steps_processed']
            del dict_shared
            agent.steps_interact, agent.step_last_record_ts = steps_interact, steps_interact # for the lambda and the logging
            evaluate_agent_mp(lambda : func_env(args), agent, num_episodes=20, type_env=type_env, step_record=None, queue_envs=queue_envs_eval, heuristic='random')
            if type_env == 'minigrid':
                agent.steps_interact, agent.step_last_record_ts = steps_interact, steps_interact
                evaluate_agent_mp(lambda : func_env(args, lava_density_range=[0.2, 0.3]), agent, num_episodes=10, type_env=type_env, step_record=None, heuristic='random', suffix='diff_0.25', record_ts=False) # diff 0.25
                agent.steps_interact, agent.step_last_record_ts = steps_interact, steps_interact
                evaluate_agent_mp(lambda : func_env(args, lava_density_range=[0.4, 0.5]), agent, num_episodes=10, type_env=type_env, step_record=None, heuristic='random', suffix='diff_0.45', record_ts=False) # diff 0.45
                agent.steps_interact, agent.step_last_record_ts = steps_interact, steps_interact
                evaluate_agent_mp(lambda : func_env(args, lava_density_range=[0.5, 0.6]), agent, num_episodes=10, type_env=type_env, step_record=None, heuristic='random', suffix='diff_0.55', record_ts=False) # diff 0.55
            if not args.ignore_model and args.step_plan_max:
                agent.steps_interact, agent.step_last_record_ts = steps_interact, steps_interact
                evaluate_agent_mp(lambda : func_env(args), agent, num_episodes=20, suffix='_best', type_env=type_env, step_record=None, queue_envs=queue_envs_eval, heuristic='best_first') # diff 0.35
                agent.steps_interact, agent.step_last_record_ts = steps_interact, steps_interact
                evaluate_agent_mp(lambda : func_env(args), agent, num_episodes=20, suffix='_modelfree', disable_planning=True, type_env=type_env, step_record=None, queue_envs=queue_envs_eval) # diff 0.35

def run_multiprocess(args, func_env_train, func_env_eval):
    pid_main = os.getpid()
    env = func_env_train(args)
    global_rb, kwargs_local_rb, queues, queue_envs_train, queue_envs_eval, event_terminate, steps_interact, episodes_interact, signal_explore, writer = prepare_experiment(env, args)
    tasks = []
    tasks.append(Process(target=generator_env, args=[queue_envs_train, queue_envs_eval, func_env_train, func_env_eval, event_terminate, args]))
    tasks.append(Process(target=explorer, args=[global_rb, kwargs_local_rb, queues[0], queue_envs_train, steps_interact, episodes_interact, event_terminate, signal_explore, args, func_env_train, writer]))
    for i in range(1, args.num_explorers):
        tasks.append(Process(target=explorer, args=[global_rb, kwargs_local_rb, queues[i], queue_envs_train, steps_interact, episodes_interact, event_terminate, signal_explore, args, func_env_train, None])) 
    tasks.append(Process(target=learner, args=[global_rb, queues, steps_interact, episodes_interact, event_terminate, signal_explore, args, pid_main, func_env_train, writer]))
    tasks.append(Process(target=evaluator, args=[steps_interact, event_terminate, queues[-1], queue_envs_eval, args, func_env_eval, writer]))
    for task in tasks: task.start()
    for task in tasks: task.join()
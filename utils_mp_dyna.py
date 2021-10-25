"""
COMPONENTS FOR DYNA BASELINE EXPERIMENTS W/ MULTI-PROCESSING
"""

import time, datetime, numpy as np
from DQN_Dyna import get_DQN_Dyna_BASE_agent, get_DQN_Dyna_agent
from utils import *
from runtime import get_cpprb_env_dict
from multiprocessing import Process, Value, Event
from multiprocessing.managers import SyncManager
from cpprb import ReplayBuffer, MPReplayBuffer, MPPrioritizedReplayBuffer
from utils import *
import os, psutil, copy
from tensorboardX import SummaryWriter

from utils_mp import import_tf, generator_env, explorer, evaluator, get_default_rb_dict
from utils import from_categorical, obs2tensor

try:
    from gym.envs.registration import register as gym_register
    gym_register(id="RandDistShift-v0", entry_point="RandDistShift:RandDistShift", reward_threshold=0.95)
except:
    pass

def explorer_dyna(global_rb_imagined, kwargs_local, queue, queue_envs_train, steps_interact, episodes_interact, event_terminate, signal_explore, args, func_env, writer, learn_model=True):
    if args.gpu_explorer:
        tf = import_tf()
    else:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    if learn_model:
        kwargs_local["env_dict"].pop("next_obs")
        kwargs_local["env_dict"].pop("rew")
        kwargs_local["env_dict"].pop("done")
    local_rb = ReplayBuffer(**kwargs_local)
    env = func_env(args)
    agent = get_DQN_Dyna_BASE_agent(env, args, writer=writer)
    agent.initialize(env.reset(), env.action_space.sample())
    size_submit = 32
    if 'procgen' in args.type_extractor.lower():
        type_env = 'procgen'
    elif 'minigrid' in args.game.lower() or 'distshift' in args.game.lower():
        type_env = 'minigrid'
    elif 'atari' in args.game.lower():
        type_env = 'atari'
    else:
        raise NotImplementedError
    flag_newenvs = args.env_pipeline and 'randdistshift' in args.game.lower()
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
            action = agent.decide(obs_curr, eval=False, env=env if type_env == 'minigrid' else None)
            obs_next, reward, done, info = env.step(action) # take a computed action
            steps_episode += 1
            if type_env == 'procgen':
                real_done = done and steps_episode != env.spec.max_episode_steps and reward == 0 and not info['prev_level_complete']
            elif type_env == 'minigrid':
                real_done = done and steps_episode != env.unwrapped.max_steps
            else:
                real_done = done
            agent.step(obs_curr, action, reward, obs_next, real_done, update=False)
            if learn_model:
                local_rb.add(obs=obs_curr, act=action)
            else:
                local_rb.add(obs=obs_curr, act=action, rew=reward, done=real_done, next_obs=obs_next)
            return_cum += reward
            obs_curr = obs_next
            flag_reset = real_done or (done and type_env == 'minigrid')
            if local_rb.get_stored_size() >= size_submit:
                if flag_reset: local_rb.on_episode_end()
                samples_local = local_rb.get_all_transitions()
                local_rb.clear()
                if learn_model:
                    obses_curr, actions = samples_local['obs'], samples_local['act']
                    obses_curr = tf.cast(obs2tensor(obses_curr), tf.float32)
                    obses_imagined, reward_dist_imagined, term_logits_imagined = agent.model(obses_curr, tf.squeeze(tf.constant(actions)), eval=True)
                    term_imagined = tf.math.argmax(term_logits_imagined, axis=-1, output_type=tf.int32)
                    reward_imagined = from_categorical(reward_dist_imagined, value_min=agent.model.predictor_reward_term.value_min, value_max=agent.model.predictor_reward_term.value_max, atoms=agent.model.predictor_reward_term.atoms, transform=agent.model.predictor_reward_term.transform)
                    samples_local['rew'], samples_local['done'], samples_local['next_obs'] = reward_imagined.numpy().reshape(-1, 1), term_imagined.numpy().reshape(-1, 1), tf.cast(tf.math.round(tf.clip_by_value(obses_imagined, clip_value_min=0, clip_value_max=96)), tf.uint8).numpy()
                if args.prioritized_replay:
                    global_rb_imagined.add(**samples_local, priorities=agent.calculate_priorities(samples_local))
                else:
                    global_rb_imagined.add(**samples_local)
        if writer is not None:
            writer.add_scalar('Performance/train', return_cum, steps_interact_curr)
            writer.add_scalar('Other/episodes', episodes_interact_curr, steps_interact_curr)
        with episodes_interact.get_lock(): episodes_interact.value += 1
        if flag_newenvs:
            del env
            if queue_envs_train.empty():
                env = func_env(args)
            else:
                env = queue_envs_train.get_nowait()

def learner_dyna(global_rb, global_rb_imagined, queues, steps_interact, episodes_interact, event_terminate, signal_explore, args, pid_main, func_env, writer):
    tf = import_tf()
    process_main = psutil.Process(pid_main)
    process_learner = psutil.Process(os.getpid())
    env = func_env(args)
    agent = get_DQN_Dyna_agent(env, args, writer=writer, replay_buffer=global_rb, replay_buffer_imagined=global_rb_imagined)

    step_last_sync, episode_last_eval, time_last_disp = 0, 0, time.time()
    print('[LEARNER] loop enter')
    agent.steps_interact = steps_interact.value
    freq_sync = 64
    flag_updated_since_sync = False
    batch_preload, batch_preload_imagined = None, None
    steps_processed_last_disp, episode_last_disp, time_last_disp = 0, 0, time.time()
    while not event_terminate.is_set():
        episodes_interact_curr = episodes_interact.value
        flag_eval = agent.initialized and (episodes_interact_curr - episode_last_eval) >= args.freq_eval
        agent.steps_interact = steps_interact.value
        flag_sync = agent.initialized and (agent.steps_interact - step_last_sync) >= freq_sync and agent.steps_interact >= agent.time_learning_starts
        flag_need_update = agent.need_update()
        if flag_need_update:
            with signal_explore.get_lock(): signal_explore.value = False
            agent.step_update(batch=batch_preload, batch_imagined=batch_preload_imagined)
            batch_preload, batch_preload_imagined = None, None
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
        elif agent.initialized:
            if batch_preload is None and global_rb.get_stored_size() >= agent.size_batch: batch_preload = agent.sample_batch()
            if batch_preload_imagined is None and global_rb_imagined.get_stored_size() >= agent.size_batch: batch_preload_imagined = agent.sample_batch(imagined=True)
        if (flag_sync and not flag_need_update and flag_updated_since_sync) or flag_eval:
            if args.method != 'DQN_Dyna' and agent.ignore_model or args.method == 'DQN_Dyna' and not args.learn_dyna_model:
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
    kwargs_imagined = copy.deepcopy(kwargs)
    kwargs_imagined['size'] = 1024
    if args.prioritized_replay:
        global_rb_imagined = MPPrioritizedReplayBuffer(**kwargs_imagined)
    else:
        global_rb_imagined = MPReplayBuffer(**kwargs_imagined)
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
    return global_rb, global_rb_imagined, kwargs_local, queues, queue_envs_train, queue_envs_eval, event_terminate, steps_interact, episodes_interact, signal_explore, glboal_writer


def run_multiprocess(args, func_env_train, func_env_eval):
    # TODO: separate the normal explorers and the dyna explorers
    pid_main = os.getpid()
    env = func_env_train(args)
    global_rb, global_rb_imagined, kwargs_local_rb, queues, queue_envs_train, queue_envs_eval, event_terminate, steps_interact, episodes_interact, signal_explore, writer = prepare_experiment(env, args)
    tasks = []
    tasks.append(Process(target=generator_env, args=[queue_envs_train, queue_envs_eval, func_env_train, func_env_eval, event_terminate, args]))
    tasks.append(Process(target=explorer, args=[global_rb, kwargs_local_rb, queues[0], queue_envs_train, steps_interact, episodes_interact, event_terminate, signal_explore, args, func_env_train, writer]))
    for i in range(1, int(args.num_explorers / 2)):
        tasks.append(Process(target=explorer, args=[global_rb, kwargs_local_rb, queues[i], queue_envs_train, steps_interact, episodes_interact, event_terminate, signal_explore, args, func_env_train, None])) 
    for i in range(int(args.num_explorers / 2), args.num_explorers):
        tasks.append(Process(target=explorer_dyna, args=[global_rb_imagined, kwargs_local_rb, queues[i], queue_envs_train, steps_interact, episodes_interact, event_terminate, signal_explore, args, func_env_train, None, args.learn_dyna_model])) 
    tasks.append(Process(target=learner_dyna, args=[global_rb, global_rb_imagined, queues, steps_interact, episodes_interact, event_terminate, signal_explore, args, pid_main, func_env_train, writer]))
    tasks.append(Process(target=evaluator, args=[steps_interact, event_terminate, queues[-1], queue_envs_eval, args, func_env_eval, writer]))
    for task in tasks: task.start()
    for task in tasks: task.join()
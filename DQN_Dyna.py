"""
DEFINITIONS OF DYNA AND DYNA* AGENT
"""

import tensorflow as tf, numpy as np
from runtime import obs2tensor, get_cpprb
from utils import to_categorical, from_categorical, embed_pos_hd, mask_change_minigrid, LinearSchedule, clip_gradients
from components import RL_AGENT, EXTRACTOR_FEATURE
from components_CP import OBJECT_EXTRACTOR, ESTIMATOR_VALUE, MODEL_TRANSITION_MINIGRIDOBS

class DQN_Dyna_NETWORK(tf.keras.Model):
    def __init__(self, extractor, head_value, **kwargs):
        super(DQN_Dyna_NETWORK, self).__init__(**kwargs)
        self.extractor, self.head_value = extractor, head_value

    @tf.function
    def __call__(self, obs, eval=False):
        u = self.extractor(obs)
        return self.head_value(u, eval=eval)

def get_DQN_Dyna_BASE_agent(env, args, writer=None):
    extractor_feature_policy = EXTRACTOR_FEATURE(shape_input=env.observation_space.shape, type_extractor=args.type_extractor, channels_out=args.len_feature, features_learnable=args.extractor_learnable)
    embed_pos, dim_additional = embed_pos_hd([extractor_feature_policy.convh, extractor_feature_policy.convw], len_embed_pos=args.len_embed_pos)
    embed_pos = tf.Variable(embed_pos, trainable=True, dtype=tf.float32)
    extractor_object_policy = OBJECT_EXTRACTOR(extractor_feature_policy, len_feature=args.len_feature, norm=args.layernorm)
    head_value_policy = ESTIMATOR_VALUE(len_feature=args.len_feature, embed_pos=embed_pos, num_actions=env.action_space.n, value_min=args.value_min, value_max=args.value_max, norm=args.layernorm, atoms=args.atoms_value, transform=args.transform_value, n_head=args.n_head)
    if args.disable_bottleneck: args.size_bottleneck = extractor_object_policy.m
    if args.learn_dyna_model:
        model_dynamics = MODEL_TRANSITION_MINIGRIDOBS(len_feature=args.len_feature, embed_pos=embed_pos, n_action_space=env.action_space.n, len_action=args.len_ebd_action, n_head=args.n_head, layers_model=args.layers_model, m=extractor_object_policy.m, n=args.size_bottleneck, reward_min=args.reward_min, reward_max=args.reward_max, atoms_reward=args.atoms_reward, norm=args.layernorm, transform_reward=args.transform_reward, QKV_depth=args.QKV_depth, QKV_width=args.QKV_width, FC_depth=args.FC_depth, FC_width=args.FC_width, type_attention=args.type_attention)
    else:
        model_dynamics = None
    return DQN_Dyna_BASE(env, extractor_object_policy, head_value_policy, model_dynamics, step_plan_max=args.step_plan_max, gamma=args.gamma, steps_total=args.steps_max, embed_pos=embed_pos, writer=writer, value_min=args.value_min, value_max=args.value_max, transform_value=args.transform_value, reward_min=args.reward_min, reward_max=args.reward_max, transform_reward=args.transform_reward, disable_debug=args.performance_only)

class DQN_Dyna_BASE(RL_AGENT):
    def __init__(self,
        env, extractor_object_policy, head_value_policy, model,
        step_plan_max=5, gamma=0.99, exploration_fraction=0.02, exploration_final_eps=0.01, epsilon_eval=0.001, steps_total=50000000, freq_record=512, clip_reward=False, embed_pos=None, writer=None, value_min=None, value_max=None, transform_value=False, reward_min=None, reward_max=None, transform_reward=False, disable_debug=False):

        super(DQN_Dyna_BASE, self).__init__(env, gamma, writer)
        self.embed_pos = embed_pos
        self.step_plan_max = step_plan_max
        if self.step_plan_max:
            self.steps_plan = LinearSchedule(schedule_timesteps=int(self.step_plan_max * 1e6), initial_p=step_plan_max, final_p=step_plan_max)
        self.epsilon = LinearSchedule(schedule_timesteps=int(exploration_fraction * steps_total), initial_p=1.0, final_p=exploration_final_eps)
        self.epsilon_eval = epsilon_eval
        
        ## policy network
        self.extractor_policy = extractor_object_policy
        self.head_value_policy = head_value_policy
        self.network_policy = DQN_Dyna_NETWORK(self.extractor_policy, self.head_value_policy, name='network_policy')
        ## model
        self.model = model
        
        self.clip_reward = bool(clip_reward)
        self.freq_record = freq_record if not disable_debug else int(1e7)
        self.steps_interact, self.steps_total = 0, steps_total
        self.obs2tensor = lambda x: obs2tensor(x, self.extractor_policy.divisor_feature, self.extractor_policy.dtype_converted_obs)
        self.step_last_record_ts = 0

        self.value_min, self.value_max, self.reward_min, self.reward_max = value_min, value_max, reward_min, reward_max
        self.transform_value, self.transform_reward = transform_value, transform_reward

    def step(self, obs_curr, action, reward, obs_next, done, update=False):
        if not self.initialized: self.initialize(obs_curr, action)

    def decide(self, obs, eval=False, env=None, disable_planning=True, suffix_record='', heuristic='', record_ts=False):
        epsilon = self.epsilon_eval if eval else self.epsilon.value(self.steps_interact)
        if np.random.rand() > epsilon:
            return int(self._decide_model_free(self.obs2tensor(obs)))
        else: # explore
            return self.action_space.sample()

    @tf.function
    def _decide_model_free(self, obs):
        return tf.math.argmax(from_categorical(self.network_policy(obs, eval=True), value_min=self.value_min, value_max=self.value_max, atoms=self.head_value_policy.atoms, transform=False), axis=-1, output_type=tf.int32) # no need to transform back, only needs the argmax

    @tf.function
    def _process_samples(self, batch_reward, batch_done):
        if self.clip_reward:
            batch_reward = tf.math.sign(batch_reward)
        else:
            batch_reward = tf.clip_by_value(batch_reward, clip_value_min=self.reward_min, clip_value_max=self.reward_max)
        batch_done = tf.cast(batch_done, tf.bool)
        batch_not_done = tf.logical_not(batch_done)
        return batch_reward, batch_done, batch_not_done

    @tf.function
    def _construct_targets_no_dist(self, batch_reward, batch_not_done, batch_obs_next):
        batch_features_next = self.extractor_policy(batch_obs_next)
        Q_next = from_categorical(self.head_value_policy(batch_features_next, eval=True), value_min=self.value_min, value_max=self.value_max, atoms=self.head_value_policy.atoms, transform=self.transform_value)
        V_next = tf.reduce_max(Q_next, axis=-1)
        target_update = tf.clip_by_value(batch_reward + self.gamma * tf.cast(batch_not_done, tf.float32) * V_next, clip_value_min=self.value_min, clip_value_max=self.value_max)
        return target_update

    @tf.function
    def _compute_priorities(self, batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done):
        batch_action, batch_reward, batch_done = tf.squeeze(batch_action), tf.squeeze(batch_reward), tf.squeeze(batch_done)
        batch_reward, batch_done, batch_not_done = self._process_samples(batch_reward, batch_done)
        target_update = self._construct_targets_no_dist(batch_reward, batch_not_done, batch_obs_next)
        batch_features_curr = self.extractor_policy(batch_obs_curr)
        Q_logits_curr = self.head_value_policy(batch_features_curr, softmax=False, eval=True)
        indices = tf.stack([tf.range(batch_action.shape[0], dtype=tf.int32), batch_action], 1)
        V_dist_curr = tf.nn.softmax(tf.gather_nd(Q_logits_curr, indices), axis=-1)
        error_TD_L1 = tf.math.abs(target_update - from_categorical(V_dist_curr, value_min=self.value_min, value_max=self.value_max, atoms=self.head_value_policy.atoms, transform=self.transform_value))
        return error_TD_L1

    def calculate_priorities(self, batch):
        batch_obs_curr, batch_action, batch_reward, batch_done, batch_obs_next = batch.values()
        batch_reward = tf.constant(batch_reward, dtype=tf.float32)
        batch_done = tf.constant(batch_done, dtype=tf.int32)
        batch_action = tf.constant(batch_action, dtype=tf.int32)
        batch_obs_curr, batch_obs_next = self.obs2tensor(batch_obs_curr), self.obs2tensor(batch_obs_next)
        error_TD_L1 = self._compute_priorities(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done)
        return error_TD_L1.numpy()

    def initialize(self, obs_curr, action):
        obs_curr = self.obs2tensor(obs_curr)
        action = tf.constant([action])
        self.extractor_policy(obs_curr)
        self.network_policy(obs_curr)
        if self.model is not None: self.model(tf.cast(obs_curr, tf.float32), action)
        self.initialized = True
    
    def weights_copyfrom(self, dict_shared):
        try:
            tf.keras.backend.set_value(self.embed_pos, dict_shared.pop('embed_pos_src'))
            self.network_policy.set_weights(dict_shared.pop('network_policy_src'))
            if self.model is not None: self.model.set_weights(dict_shared.pop('model_src'))
        except:
            print('dict_shared is None: skipped parameter sync')
        return dict_shared

def get_DQN_Dyna_agent(env, args, replay_buffer=None, replay_buffer_imagined=None, writer=None):
    if replay_buffer is None: replay_buffer = get_cpprb(env, args.size_buffer, prioritized=args.prioritized_replay)
    if replay_buffer_imagined is None: replay_buffer_imagined = get_cpprb(env, args.size_buffer, prioritized=args.prioritized_replay)
    extractor_feature_policy = EXTRACTOR_FEATURE(shape_input=env.observation_space.shape, type_extractor=args.type_extractor, channels_out=args.len_feature, features_learnable=args.extractor_learnable)
    embed_pos, dim_additional = embed_pos_hd([extractor_feature_policy.convh, extractor_feature_policy.convw], len_embed_pos=args.len_embed_pos)
    embed_pos = tf.Variable(embed_pos, trainable=True, dtype=tf.float32)
    extractor_object_policy = OBJECT_EXTRACTOR(extractor_feature_policy, len_feature=args.len_feature, norm=args.layernorm)
    head_value_policy = ESTIMATOR_VALUE(len_feature=args.len_feature, embed_pos=embed_pos, num_actions=env.action_space.n, value_min=args.value_min, value_max=args.value_max, norm=args.layernorm, atoms=args.atoms_value, transform=args.transform_value, n_head=args.n_head)
    extractor_feature_target = EXTRACTOR_FEATURE(shape_input=env.observation_space.shape, type_extractor=args.type_extractor, channels_out=args.len_feature)
    extractor_object_target = OBJECT_EXTRACTOR(extractor_feature_target, len_feature=args.len_feature, norm=args.layernorm)
    extractor_object_target.trainable = False
    head_value_target = ESTIMATOR_VALUE(len_feature=args.len_feature, embed_pos=embed_pos, num_actions=env.action_space.n, value_min=args.value_min, value_max=args.value_max, norm=args.layernorm, atoms=args.atoms_value, transform=args.transform_value, n_head=args.n_head)
    head_value_target.trainable = False
    if args.disable_bottleneck: args.size_bottleneck = extractor_object_policy.m
    if args.learn_dyna_model:
        model_dynamics = MODEL_TRANSITION_MINIGRIDOBS(len_feature=args.len_feature, embed_pos=embed_pos, n_action_space=env.action_space.n, len_action=args.len_ebd_action, n_head=args.n_head, layers_model=args.layers_model, m=extractor_object_policy.m, n=args.size_bottleneck, reward_min=args.reward_min, reward_max=args.reward_max, atoms_reward=args.atoms_reward, norm=args.layernorm, transform_reward=args.transform_reward, QKV_depth=args.QKV_depth, QKV_width=args.QKV_width, FC_depth=args.FC_depth, FC_width=args.FC_width, type_attention=args.type_attention)
    else:
        model_dynamics = None
    return DQN_Dyna(env, extractor_object_policy, extractor_object_target, head_value_policy, head_value_target, model_dynamics, replay_buffer, replay_buffer_imagined, size_bottleneck=args.size_bottleneck, size_batch=args.size_batch, clip_reward=args.clip_reward, steps_total=args.steps_max, prioritized_replay=args.prioritized_replay, ignore_TD=args.ignore_TD, type_optimizer=args.type_optimizer, step_plan_max=args.step_plan_max, gpu_buffer=args.gpu_buffer, gamma=args.gamma, lr=args.lr, freq_train_TD=args.freq_train_TD, freq_train_model=args.freq_train_model, embed_pos=embed_pos, writer=writer, value_min=args.value_min, value_max=args.value_max, transform_value=args.transform_value, reward_min=args.reward_min, reward_max=args.reward_max, transform_reward=args.transform_reward, disable_debug=args.performance_only)

class DQN_Dyna(DQN_Dyna_BASE):
    def __init__(self,
        env,
        extractor_object_policy, extractor_object_target,
        head_value_policy, head_value_target,
        model,
        replay_buffer, replay_buffer_imagined,
        size_bottleneck=16,
        step_plan_max=5,
        gamma=0.99,
        exploration_fraction=0.02, exploration_final_eps=0.01, epsilon_eval=0.001, steps_total=50000000,
        prioritized_replay=True,
        lr=0.0000625, eps=1.5e-4,
        freq_targetnet_update=8000, freq_train_TD=4, freq_train_model=4, size_batch=32,
        type_optimizer='Adam',
        clip_gradient_TD=True, clip_gradient_model=True,
        clip_reward=False, ignore_TD=False, gpu_buffer=False, embed_pos=None, writer=None,
        value_min=None, value_max=None, transform_value=False, reward_min=None, reward_max=None, transform_reward=False, disable_debug=False):

        super(DQN_Dyna, self).__init__(env, extractor_object_policy, head_value_policy, model,
        step_plan_max=step_plan_max, gamma=gamma, exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps, epsilon_eval=epsilon_eval, steps_total=steps_total, embed_pos=embed_pos, clip_reward=clip_reward, writer=writer, value_min=value_min, value_max=value_max, transform_value=transform_value, reward_min=reward_min, reward_max=reward_max, transform_reward=transform_reward, disable_debug=disable_debug)

        self.replay_buffer, self.replay_buffer_imagined = replay_buffer, replay_buffer_imagined
        self.clip_gradient_TD, self.clip_gradient_model = bool(clip_gradient_TD), bool(clip_gradient_model)
        self.ignore_TD = bool(ignore_TD)
        self.gpu_buffer = bool(gpu_buffer)

        ## target network
        self.extractor_target = extractor_object_target
        self.head_value_target = head_value_target
        if self.extractor_target is not None and self.head_value_target is not None:
            self.network_target = DQN_Dyna_NETWORK(self.extractor_target, self.head_value_target, name='network_target')
            self.network_target.trainable = False

        if type_optimizer == 'Adam':
            if not self.ignore_TD: self.optimizer_TD = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps)
            if self.model is not None: self.optimizer_model = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps)
        elif type_optimizer == 'RMSprop':
            if not self.ignore_TD: self.optimizer_TD = tf.keras.optimizers.RMSprop(learning_rate=lr, epsilon=eps)
            if self.model is not None: self.optimizer_model = tf.keras.optimizers.RMSprop(learning_rate=lr, epsilon=eps)
        self.size_batch = size_batch
        self.size_wholeset, self.size_bottleneck = self.extractor_policy.m, size_bottleneck

        self.prioritized_replay = bool(prioritized_replay)
        self.time_learning_starts = 20000 if self.prioritized_replay else 50000

        self.freq_train_TD, self.freq_train_model = freq_train_TD, freq_train_model
        self.freq_targetnet_update = freq_targetnet_update
        self.flag_optimizers_initialized = False
        self.step_last_targetnet_update, self.step_last_update_record = self.time_learning_starts - self.freq_targetnet_update, self.time_learning_starts - self.freq_record
        self.step_last_update_TD = np.inf if self.ignore_TD else self.time_learning_starts - self.freq_train_TD
        self.step_last_update_model = self.time_learning_starts - self.freq_train_model
        self.steps_processed = self.step_last_update_TD if self.model is None else min(self.step_last_update_TD, self.step_last_update_model)

    def step(self, obs_curr, action, reward, obs_next, done, update=True): # for single process runs
        if obs_next is not None: self.replay_buffer.add(obs=obs_curr, act=action, rew=reward, next_obs=obs_next, done=done) # Dyna does not run with single process, so this is depracated
        if update: self.step_update()
        self.steps_interact += 1

    def step_update(self, batch=None, batch_imagined=None):
        if not self.initialized and self.replay_buffer.get_stored_size() >= self.size_batch: self.initialize()
        if self.steps_interact >= self.time_learning_starts and self.replay_buffer.get_stored_size() >= self.size_batch and self.replay_buffer_imagined.get_stored_size() >= self.size_batch:
            flag_train_TD = not self.ignore_TD and (self.steps_interact - self.step_last_update_TD) >= self.freq_train_TD
            flag_train_model = self.model is not None and (self.steps_interact - self.step_last_update_model) >= self.freq_train_model
            if flag_train_TD or flag_train_model:
                self.update(flag_train_TD, flag_train_model, batch=batch, batch_imagined=batch_imagined)
            if not self.ignore_TD and (self.steps_interact - self.step_last_targetnet_update) >= self.freq_targetnet_update:
                self.sync_parameters()
                self.step_last_targetnet_update += self.freq_targetnet_update
        self.steps_processed = self.step_last_update_TD if self.model is None else min(self.step_last_update_TD, self.step_last_update_model)

    def need_update(self):
        if not self.initialized and self.replay_buffer.get_stored_size() >= self.size_batch: return True
        if self.steps_interact >= self.time_learning_starts and self.replay_buffer.get_stored_size() >= self.size_batch and self.replay_buffer_imagined.get_stored_size() >= self.size_batch:
            flag_train_TD = not self.ignore_TD and (self.steps_interact - self.step_last_update_TD) >= self.freq_train_TD
            flag_train_model = self.model is not None and (self.steps_interact - self.step_last_update_model) >= self.freq_train_model
            if flag_train_TD or flag_train_model: return True
            if not self.ignore_TD and (self.steps_interact - self.step_last_targetnet_update) >= self.freq_targetnet_update:
                return True
        return False

    @tf.function
    def _apply_gradients_TD(self, gradients_TD, clip_TD=True):
        if gradients_TD is not None:
            if clip_TD: gradients_TD = clip_gradients(gradients_TD)
            self.optimizer_TD.apply_gradients(zip(gradients_TD, self.parameters_train_TD))
        else:
            gradients_TD = None
        return gradients_TD

    @tf.function
    def _apply_gradients_model(self, gradients_model, clip_model=True):
        if gradients_model is not None:
            if clip_model: gradients_model = clip_gradients(gradients_model)
            self.optimizer_model.apply_gradients(zip(gradients_model, self.parameters_train_model))
        else:
            gradients_model = None
        return gradients_model

    @tf.function
    def _update_TD_Dyna(self, batch_obs_curr, batch_obs_curr_imagined, batch_action, batch_action_imagined, batch_reward, batch_reward_imagined, batch_obs_next, batch_obs_next_imagined, batch_not_done, batch_not_done_imagined, weights, weights_imagined, flag_record=False):
        batch_obs_curr_combined, batch_action_combined, batch_reward_combined, batch_obs_next_combined, batch_not_done_combined = tf.concat([batch_obs_curr, batch_obs_curr_imagined], axis=0), tf.concat([batch_action, batch_action_imagined], axis=0), tf.concat([batch_reward, batch_reward_imagined], axis=0), tf.concat([batch_obs_next, batch_obs_next_imagined], axis=0), tf.concat([batch_not_done, batch_not_done_imagined], axis=0)
        if self.prioritized_replay:
            weights_combined = tf.concat([weights, weights_imagined], axis=0)
        else:
            weights_combined = None
        gradients_TD, error_TD_weighted, error_TD_L1_combined, error_TD_L1_weighted, _, _ = self._update_TD(batch_obs_curr_combined, batch_obs_next_combined, batch_action_combined, batch_reward_combined, batch_not_done_combined, weights_combined, flag_record=flag_record)
        error_TD_L1, error_TD_L1_imagined = tf.split(error_TD_L1_combined, 2, axis=0)
        return gradients_TD, error_TD_weighted, error_TD_L1, error_TD_L1_imagined, error_TD_L1_weighted

    def update(self, flag_train_TD=True, flag_train_model=True, batch=None, batch_imagined=None):
        flag_record = (self.steps_interact - self.step_last_update_record) >= self.freq_record
        batch_features_next_policy, Q_dist_next_policy = None, None
        if not self.flag_optimizers_initialized or flag_train_TD:
            if batch is None: batch = self.sample_batch()
            batch_obs_curr, batch_action, batch_reward, batch_obs_next, _, batch_not_done, weights, batch_idxes = batch
            if batch_imagined is None: batch_imagined = self.sample_batch(imagined=True)
            batch_obs_curr_imagined, batch_action_imagined, batch_reward_imagined, batch_obs_next_imagined, _, batch_not_done_imagined, weights_imagined, batch_idxes_imagined = batch_imagined
            gradients_TD, error_TD_weighted, error_TD_L1, error_TD_L1_imagined, error_TD_L1_weighted = self._update_TD_Dyna(batch_obs_curr, batch_obs_curr_imagined, batch_action, batch_action_imagined, batch_reward, batch_reward_imagined, batch_obs_next, batch_obs_next_imagined, batch_not_done, batch_not_done_imagined, weights, weights_imagined, flag_record=flag_record)
            ## update prioritized replay, if used
            if self.prioritized_replay:
                self.replay_buffer.update_priorities(batch_idxes, error_TD_L1.numpy())
                self.replay_buffer_imagined.update_priorities(batch_idxes_imagined, error_TD_L1_imagined.numpy())
            if flag_record:
                self.record_scalar('Error/TD', error_TD_weighted, self.step_last_update_TD)
                self.record_scalar('Debug/norm_gradient_TD', tf.linalg.global_norm(gradients_TD), self.step_last_update_TD)
                self.record_scalar('Debug/TD_L1', error_TD_L1_weighted, self.step_last_update_TD)
        else:
            error_TD_weighted, gradients_TD = 0, None
        ## model gradients!
        if not flag_train_model:
            if not flag_train_TD:
                if flag_record:
                    self.step_last_update_record += self.freq_record
                return
            else:
                self._apply_gradients_TD(gradients_TD, clip_TD=self.clip_gradient_TD)
                self.step_last_update_TD += self.freq_train_TD
        else:
            if batch is None: batch = self.sample_batch()
            batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_not_done, weights, batch_idxes = batch
            gradients_model, error_reward_imagined_L1_weighted, tp_term_imagined, fn_term_imagined, acc_term_imagined, error_dynamics_L1_changed_relative, error_dynamics_L1_unchanged_relative, error_model_weighted, error_dynamics_weighted, error_term_imagined_weighted, error_reward_imagined_weighted, cosdist_features, norm_features_L1, error_dynamics_L1_relative = self._update_model(batch_obs_curr, batch_action, batch_obs_next, batch_done, batch_not_done, batch_reward, batch_features_next_policy, Q_dist_next_policy, weights, flag_record=flag_record)
            if flag_train_TD:
                self._apply_gradients_TD(gradients_TD, clip_TD=self.clip_gradient_TD)
                self.step_last_update_TD += self.freq_train_TD
            self._apply_gradients_model(gradients_model, clip_model=self.clip_gradient_model)
            self.step_last_update_model += self.freq_train_model
            if not self.flag_optimizers_initialized: self.flag_optimizers_initialized = True
            if flag_record:
                # if not self.ignore_TD and gradients_TD_clip is not None: self.record_scalar('Debug/norm_gradient_TD_clipped', tf.linalg.global_norm(gradients_TD_clip), self.step_last_update_TD)
                if flag_train_model:
                    if self.extractor_policy.features_learnable:
                        self.record_scalar('Debug/norm_features_L1', norm_features_L1, self.step_last_update_model)
                        self.record_scalar('Debug/cosdist_features', cosdist_features, self.step_last_update_model)
                    self.record_scalar('Error/dynamics', error_dynamics_weighted, self.step_last_update_model)
                    self.record_scalar('Error/reward_imagined', error_reward_imagined_weighted, self.step_last_update_model) # reward predicted using the imagined next observation
                    self.record_scalar('Error/term_imagined', error_term_imagined_weighted, self.step_last_update_model) # term predicted using the imagined next observation
                    self.record_scalar('Error/overall', error_TD_weighted + error_model_weighted, self.step_last_update_model)
                    self.record_scalar('Debug/norm_gradient_model', tf.linalg.global_norm(gradients_model), self.step_last_update_model)
                    # if gradients_model_clip is not None: self.record_scalar('Debug/norm_gradient_model_clipped', tf.linalg.global_norm(gradients_model_clip), self.step_last_update_model)
                    self.record_scalar('Debug/reward_imagined_L1', error_reward_imagined_L1_weighted, self.step_last_update_model)
                    self.record_scalar('Debug/error_dynamics_L1_relative', error_dynamics_L1_relative, self.step_last_update_model)
                    # self.record_scalar('Debug/error_dynamics_L1_elementwise', error_dynamics_L1_elementwise, self.step_last_update_model)
                    if error_dynamics_L1_changed_relative is not None: self.record_scalar('Debug/error_dynamics_L1_changed_relative', error_dynamics_L1_changed_relative, self.step_last_update_model)
                    if error_dynamics_L1_unchanged_relative is not None: self.record_scalar('Debug/error_dynamics_L1_unchanged_relative', error_dynamics_L1_unchanged_relative, self.step_last_update_model)
                    if tp_term_imagined is not None and not bool(tf.math.is_nan(tp_term_imagined)): self.record_scalar('Debug/tp_term_imagined', tp_term_imagined, self.step_last_update_model)
                    if fn_term_imagined is not None and not bool(tf.math.is_nan(fn_term_imagined)): self.record_scalar('Debug/fn_term_imagined', fn_term_imagined, self.step_last_update_model)
                    self.record_scalar('Debug/acc_term_imagined', acc_term_imagined, self.step_last_update_model)

        if flag_record:
            self.step_last_update_record += self.freq_record

    @tf.function
    def _construct_targets_DDQN(self, batch_reward, batch_not_done, batch_obs_next):
        size_batch = tf.size(batch_reward)
        batch_features_next_target = self.extractor_target(batch_obs_next)
        Q_next_target = from_categorical(self.head_value_target(batch_features_next_target, eval=True), value_min=self.value_min, value_max=self.value_max, atoms=self.head_value_policy.atoms, transform=self.transform_value)
        batch_features_next_policy = self.extractor_policy(batch_obs_next)
        Q_dist_next_policy = self.head_value_policy(batch_features_next_policy, eval=True)
        batch_action_next_policy = tf.math.argmax(from_categorical(Q_dist_next_policy, value_min=self.value_min, value_max=self.value_max, atoms=self.head_value_policy.atoms, transform=False), axis=-1, output_type=tf.int32) # no need to transform back, only needs the argmax
        V_next_target = tf.gather_nd(Q_next_target, tf.stack([tf.range(size_batch, dtype=tf.int32), batch_action_next_policy], 1))
        target_update = tf.clip_by_value(batch_reward + self.gamma * tf.cast(batch_not_done, tf.float32) * V_next_target, clip_value_min=self.value_min, clip_value_max=self.value_max)
        target_dist_update = to_categorical(target_update, value_min=self.value_min, value_max=self.value_max, atoms=self.head_value_policy.atoms, transform=self.transform_value, clip=False)
        return target_dist_update, target_update, batch_features_next_policy, Q_dist_next_policy

    @tf.function
    def _construct_targets_DQN(self, batch_reward, batch_not_done, batch_obs_next):
        batch_features_next_target = self.extractor_target(batch_obs_next)
        Q_next_target = from_categorical(self.head_value_target(batch_features_next_target, eval=True), value_min=self.value_min, value_max=self.value_max, atoms=self.head_value_policy.atoms, transform=self.transform_value)
        V_next_target = tf.reduce_max(Q_next_target, axis=-1)
        target_update = tf.clip_by_value(batch_reward + self.gamma * tf.cast(batch_not_done, tf.float32) * V_next_target, clip_value_min=self.value_min, clip_value_max=self.value_max)
        target_dist_update = to_categorical(target_update, value_min=self.value_min, value_max=self.value_max, atoms=self.head_value_policy.atoms, transform=self.transform_value, clip=False)
        return target_dist_update, target_update

    @tf.function
    def _update_TD(self, batch_obs_curr, batch_obs_next, batch_action, batch_reward, batch_not_done, weights, flag_record=True):
        target_dist_update, target_update, batch_features_next_policy, Q_dist_next_policy = self._construct_targets_DDQN(batch_reward, batch_not_done, batch_obs_next)
        indices = tf.stack([tf.range(batch_action.shape[0], dtype=tf.int32), batch_action], 1)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.parameters_train_TD)
            batch_features_curr = self.extractor_policy(batch_obs_curr)
            Q_logits_curr = self.head_value_policy(batch_features_curr, softmax=False, eval=True)
            V_dist_curr = tf.nn.softmax(tf.gather_nd(Q_logits_curr, indices), axis=-1)
            error_TD = tf.keras.losses.KLD(tf.stop_gradient(target_dist_update), V_dist_curr)
            error_TD_weighted = self.weight_error(error_TD, weights)
        gradients_TD = tape.gradient(error_TD_weighted, self.parameters_train_TD)
        if self.prioritized_replay or flag_record:
            error_TD_L1 = tf.math.abs(target_update - from_categorical(V_dist_curr, value_min=self.value_min, value_max=self.value_max, atoms=self.head_value_policy.atoms, transform=self.transform_value))
        else:
            error_TD_L1 = None
        if flag_record:
            error_TD_L1_weighted = self.weight_error(error_TD_L1, weights)
        else:
            error_TD_L1_weighted = None
        return gradients_TD, error_TD_weighted, error_TD_L1, error_TD_L1_weighted, batch_features_next_policy, Q_dist_next_policy

    @tf.function
    def _norm_cosdist_features(self, batch_features):
        batch_features_normalized, _ = tf.linalg.normalize(tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[-1])), ord=2, axis=-1)
        cosdist_features = tf.reduce_mean(tf.einsum('abx,adx->abd', batch_features_normalized, batch_features_normalized)) * self.size_wholeset / (self.size_wholeset - 1) - 1.0 / self.size_wholeset
        norm_features_L1 = tf.reduce_mean(tf.reduce_sum(tf.math.abs(batch_features), -1))
        return cosdist_features, norm_features_L1

    @tf.function
    def _update_model_forward(self, batch_obs_curr, batch_action, batch_done, batch_reward_categorical, batch_obs_next):
        obses_imagined, batch_reward_dist_imagined, batch_done_logits_imagined = self.model.forward_train(batch_obs_curr, batch_action)
        ## calculate termination imagination error
        error_term_imagined = tf.keras.losses.sparse_categorical_crossentropy(tf.stop_gradient(batch_done), batch_done_logits_imagined, from_logits=True, axis=-1)
        error_term_imagined_weighted = tf.reduce_mean(error_term_imagined)
        ## calculate reward imagination error
        error_reward_imagined = tf.keras.losses.KLD(tf.stop_gradient(batch_reward_categorical), batch_reward_dist_imagined)
        error_reward_imagined_weighted = tf.reduce_mean(error_reward_imagined)
        ## calculate dynamics consistency error
        error_dynamics_L1 = tf.math.abs(tf.stop_gradient(tf.cast(batch_obs_next, tf.float32)) - obses_imagined)
        error_dynamics_L1 = tf.reshape(error_dynamics_L1, [error_dynamics_L1.shape[0], -1, error_dynamics_L1.shape[-1]])
        error_dynamics = error_dynamics_L1 ** 2
        error_dynamics_weighted = tf.reduce_mean(error_dynamics)
        ## add up!
        error_model_weighted = error_dynamics_weighted + error_term_imagined_weighted + error_reward_imagined_weighted
        return error_model_weighted, error_term_imagined_weighted, error_reward_imagined_weighted, error_dynamics_weighted, batch_reward_dist_imagined, batch_done_logits_imagined, error_dynamics_L1

    @tf.function
    def _update_model(self, batch_obs_curr, batch_action, batch_obs_next, batch_done, batch_not_done, batch_reward, batch_features_next, Q_dist_next, weights, flag_record=False):
        index_term_trans, index_nonterm_trans = tf.squeeze(tf.where(batch_done)), tf.squeeze(tf.where(batch_not_done))
        batch_reward_categorical = to_categorical(batch_reward, value_min=self.reward_min, value_max=self.reward_max, atoms=self.model.predictor_reward_term.atoms, transform=self.transform_reward, clip=False)
        if batch_features_next is None: batch_features_next = self.extractor_policy(batch_obs_next) # if from the same batch and DDQN, this thing is already computed
        if Q_dist_next is None: Q_dist_next = self.head_value_policy(batch_features_next, eval=True) # if from the same batch and DDQN, this thing is already computed
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.parameters_train_model)
            error_model_weighted, error_term_imagined_weighted, error_reward_imagined_weighted, error_dynamics_weighted, batch_reward_dist_imagined, batch_done_logits_imagined, error_dynamics_L1 = self._update_model_forward(tf.cast(batch_obs_curr, tf.float32), batch_action, batch_done, batch_reward_categorical, batch_obs_next)
        gradients_model = tape.gradient(error_model_weighted, self.parameters_train_model)
        if flag_record:
            cosdist_features, norm_features_L1 = self._norm_cosdist_features(batch_features_next)
            norm_features_L1_elementwise = norm_features_L1 / self.extractor_policy.len_feature
            error_dynamics_L1_relative = tf.reduce_mean(error_dynamics_L1) / norm_features_L1_elementwise
            if 'minigrid' in self.extractor_policy.type_env:
                error_dynamics_L1_objectwise_relative = tf.reduce_mean(error_dynamics_L1[:, :, 0: 3], axis=-1) / norm_features_L1_elementwise
                mask_change = mask_change_minigrid(batch_obs_curr, batch_obs_next)
                error_dynamics_L1_changed_relative = tf.reduce_mean(tf.boolean_mask(error_dynamics_L1_objectwise_relative, mask_change))
                error_dynamics_L1_unchanged_relative = tf.reduce_mean(tf.boolean_mask(error_dynamics_L1_objectwise_relative, tf.math.logical_not(mask_change)))
            else:
                error_dynamics_L1_objectwise_relative = tf.reduce_mean(error_dynamics_L1, axis=-1) / norm_features_L1_elementwise
                error_dynamics_L1_changed_relative, error_dynamics_L1_unchanged_relative = None, None
            batch_reward_imagined = from_categorical(batch_reward_dist_imagined, value_min=self.reward_min, value_max=self.reward_max, atoms=self.model.predictor_reward_term.atoms, transform=self.transform_reward)
            error_reward_imagined_L1_weighted = tf.reduce_mean(tf.math.abs(batch_reward - batch_reward_imagined))
            batch_done_imagined_compact = tf.dtypes.cast(tf.argmax(batch_done_logits_imagined, axis=-1, output_type=tf.int32), tf.bool)
            eq_done = tf.dtypes.cast(batch_done == batch_done_imagined_compact, tf.float32)
            acc_term_imagined = tf.reduce_mean(eq_done)
            tp_term_imagined = tf.reduce_mean(tf.gather(eq_done, index_term_trans, axis=-1))
            fn_term_imagined = tf.reduce_mean(tf.gather(eq_done, index_nonterm_trans, axis=-1))
        else:
            cosdist_features, norm_features_L1 = None, None
            error_reward_imagined_L1_weighted = None
            acc_term_imagined, tp_term_imagined, fn_term_imagined = None, None, None
            error_dynamics_L1_changed_relative, error_dynamics_L1_unchanged_relative, error_dynamics_L1_relative = None, None, None
        return gradients_model, error_reward_imagined_L1_weighted, tp_term_imagined, fn_term_imagined, acc_term_imagined, error_dynamics_L1_changed_relative, error_dynamics_L1_unchanged_relative, error_model_weighted, error_dynamics_weighted, error_term_imagined_weighted, error_reward_imagined_weighted, cosdist_features, norm_features_L1, error_dynamics_L1_relative

    def sample_batch(self, size_batch=None, imagined=False):
        if size_batch is None: size_batch = self.size_batch
        if imagined:
            batch_samples = self.replay_buffer_imagined.sample(size_batch)
        else:
            batch_samples = self.replay_buffer.sample(size_batch)
        if self.prioritized_replay:
            batch_obs_curr, batch_action, batch_reward, batch_done, batch_obs_next, weights, batch_idxes = batch_samples.values()
            weights_tf = tf.constant(weights, dtype=tf.float32)
        else:
            batch_obs_curr, batch_action, batch_reward, batch_done, batch_obs_next = batch_samples.values()
            weights_tf, batch_idxes = None, None
        batch_reward_tf = tf.constant(batch_reward, dtype=tf.float32)
        batch_done_tf = tf.constant(batch_done, dtype=tf.int32)
        batch_action_tf = tf.constant(batch_action, dtype=tf.int32)
        batch_obs_curr_tf, batch_obs_next_tf = self.obs2tensor(batch_obs_curr), self.obs2tensor(batch_obs_next)
        batch_action_tf, batch_reward_tf, batch_done_tf = tf.squeeze(batch_action_tf), tf.squeeze(batch_reward_tf), tf.squeeze(batch_done_tf)
        batch_reward_tf, batch_done_tf, batch_not_done_tf = self._process_samples(batch_reward_tf, batch_done_tf)
        return (batch_obs_curr_tf, batch_action_tf, batch_reward_tf, batch_obs_next_tf, batch_done_tf, batch_not_done_tf, weights_tf, batch_idxes)

    def sync_parameters(self):
        self.network_target.set_weights(self.network_policy.get_weights())

    @tf.function
    def weight_error(self, error, weights):
        if self.prioritized_replay:
            return tf.tensordot(error, weights, 1)
        else:
            return tf.reduce_mean(error)

    def initialize(self):
        batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_not_done, _, _ = self.sample_batch()
        self.extractor_policy(batch_obs_curr)
        self.extractor_target(batch_obs_curr)
        self._construct_targets_DDQN(batch_reward, batch_not_done, batch_obs_next)
        self.network_policy(batch_obs_curr)
        self.network_target(batch_obs_curr)
        self.parameters_train_TD = self.network_policy.trainable_variables
        if self.model is not None:
            self.model(tf.cast(batch_obs_curr, tf.float32), batch_action)
            self.parameters_train_model = self.model.trainable_variables
        self.sync_parameters()
        self.initialized = True
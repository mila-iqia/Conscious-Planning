import tensorflow as tf
from utils import noisy_shift2, MultiHeadAttention

class OBJECT_EXTRACTOR(tf.keras.layers.Layer):
    """extracting objects from feature representations: inputs a state representation and outputs a set of object embeddings"""
    def __init__(self, feature_extractor, len_feature, norm=False):
        super(OBJECT_EXTRACTOR, self).__init__(name='extractor')
        self.feature_extractor = feature_extractor
        self.type_env, self.type_extractor = self.feature_extractor.type_env, self.feature_extractor.type_extractor
        self.convh, self.convw, self.m = self.feature_extractor.convh, self.feature_extractor.convw, self.feature_extractor.m
        self.divisor_feature, self.dtype_converted_obs, self.features_learnable = self.feature_extractor.divisor_feature, self.feature_extractor.dtype_converted_obs, self.feature_extractor.features_learnable
        self.len_feature = len_feature
        self.norm = norm
        if self.norm: self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)

    @tf.function
    def __call__(self, obs):
        x = tf.reshape(self.feature_extractor(obs), (-1, self.m, self.len_feature))
        return self.layernorm(x) if self.norm else x

class TRANSFORMER_AUGMENTED(tf.keras.layers.Layer):
    def __init__(self, len_object, num_layers, len_action=0, norm=False, n_head=8, FC_depth=3, FC_width=64):
        super(TRANSFORMER_AUGMENTED, self).__init__(name='transformer augmented')
        self.len_object, self.len_action = len_object, len_action
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(SUBLAYER_TRANSFORMER_MHA(len_object=self.len_object, n_head=n_head, norm=norm))
            if self.len_action:
                self.layers.append(SUBLAYER_TRANSFORMER_ACTION(len_object=self.len_object, num_layers=FC_depth, width=FC_width, norm=norm))
            else:
                self.layers.append(SUBLAYER_TRANSFORMER_FC(len_object=self.len_object, num_layers=FC_depth, width=FC_width, norm=norm))

    @tf.function
    def __call__(self, objects, ebd_action=None):
        if self.len_action:
            for layer_mha, layer_action in zip(self.layers[::2], self.layers[1::2]):
                objects = layer_mha(objects)
                objects = layer_action(objects, ebd_action)
        else:
            for layer in self.layers: objects = layer(objects)
        return objects

class ESTIMATOR_VALUE(tf.keras.layers.Layer):
    """ The value estimator that takes a set of objects as input and outputs the estimated state-action values """
    def __init__(self, len_feature, embed_pos, num_actions, num_layers=3, width=64, value_min=-1, value_max=1, atoms=64, norm=False, noisy_shift=False, n_head=8):
        super(ESTIMATOR_VALUE, self).__init__(name='head_value')
        self.noisy_shift = noisy_shift
        self.len_feature, self.num_actions = len_feature, num_actions
        self.value_min, self.value_max, self.atoms = float(value_min), float(value_max), int(atoms)
        self.embed_pos = embed_pos
        self.len_object = self.len_feature + embed_pos.shape[-1]
        self.layers = TRANSFORMER_AUGMENTED(len_object=self.len_object, len_action=0, num_layers=num_layers, n_head=n_head, norm=norm)
        self.dim_scaler = tf.keras.layers.Conv1D(width, kernel_size=1, activation='relu', strides=1)
        self.pooler = tf.keras.models.Sequential([
            tf.keras.layers.Dense(width, activation='relu'),
            tf.keras.layers.Dense(width, activation='relu'),
            tf.keras.layers.Dense(num_actions * self.atoms),
        ])

    @tf.function
    def __call__(self, features, softmax=True, eval=False):
        embed_pos = tf.repeat(self.embed_pos, features.shape[0], axis=0)
        if not eval and self.noisy_shift: embed_pos = noisy_shift2(embed_pos)
        objects = tf.concat([features, embed_pos], axis=-1)
        objects = self.dim_scaler(self.layers(objects))
        summary = tf.reduce_mean(objects, axis=1)
        logits = tf.reshape(self.pooler(summary), (-1, self.num_actions, self.atoms))
        if softmax:
            return tf.nn.softmax(logits, axis=-1)
        else:
            return logits

class SUBLAYER_TRANSFORMER_ACTION(tf.keras.layers.Layer):
    def __init__(self, len_object=64, num_layers=2, width=64, residual=True, norm=False):
        super(SUBLAYER_TRANSFORMER_ACTION, self).__init__('sublayer_FC_with_action')
        self.residual, self.norm = residual, norm
        if self.norm: self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)
        if num_layers == 1:
            self.fc = tf.keras.layers.Conv1D(len_object, kernel_size=1, strides=1)
        else:
            self.fc = tf.keras.models.Sequential()
            for num_layer in range(num_layers):
                if num_layer < num_layers - 1:
                    self.fc.add(tf.keras.layers.Conv1D(width, kernel_size=1, strides=1, activation='relu'))
                else:
                    self.fc.add(tf.keras.layers.Conv1D(len_object, kernel_size=1, strides=1))

    @tf.function
    def __call__(self, objects_in, action):
        increment = self.fc(tf.concat([tf.repeat(tf.expand_dims(action, 1), objects_in.shape[1], axis=1), objects_in], axis=-1))
        objects_out = objects_in + increment if self.residual else increment
        return self.layernorm(objects_out) if self.norm else objects_out

class SUBLAYER_TRANSFORMER_FC(tf.keras.layers.Layer):
    def __init__(self, len_object=64, num_layers=2, width=64, residual=True, norm=False):
        super(SUBLAYER_TRANSFORMER_FC, self).__init__('sublayer_FC')
        self.residual, self.norm = residual, norm
        if self.norm: self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)
        if num_layers == 1:
            self.fc = tf.keras.layers.Conv1D(len_object, kernel_size=1, strides=1)
        else:
            self.fc = tf.keras.models.Sequential()
            for layer in range(num_layers):
                if layer < num_layers - 1:
                    self.fc.add(tf.keras.layers.Conv1D(width, kernel_size=1, strides=1, activation='relu'))
                else:
                    self.fc.add(tf.keras.layers.Conv1D(len_object, kernel_size=1, strides=1))

    @tf.function
    def __call__(self, objects_in):
        increment = self.fc(objects_in)
        objects_out = objects_in + increment if self.residual else increment
        return self.layernorm(objects_out) if self.norm else objects_out

class SUBLAYER_TRANSFORMER_MHA(tf.keras.layers.Layer):
    def __init__(self, len_object=64, n_head=8, residual=True, norm=False):
        super(SUBLAYER_TRANSFORMER_MHA, self).__init__(name='sublayer_MHA')
        self.residual, self.norm = residual, norm
        if self.norm: self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)
        self.self_attn = MultiHeadAttention(len_object, n_head)

    @tf.function
    def __call__(self, objects_in):
        increment, _ = self.self_attn(objects_in, objects_in, objects_in)
        objects_out = objects_in + increment if self.residual else increment
        return self.layernorm(objects_out) if self.norm else objects_out

class MODEL_TRANSITION(tf.keras.Model):
    def __init__(self, n_action_space, len_action, len_feature, embed_pos, layers_model=3, n_head=8, m=64, n=4, FC_width=64, FC_depth=3, norm=False, depth_reward_term_predictor=1, reward_min=-1, reward_max=1, atoms_reward=64, noisy_shift=False, signal_predict_action=True, depth_FC_action_predictor=1, width_pool=64, type_compress='semihard'):
        super(MODEL_TRANSITION, self).__init__(name='model_transition')
        self.noisy_shift = noisy_shift
        self.len_feature, self.n_action_space, self.len_action, self.embed_pos, self.len_pos, self.len_object = len_feature, n_action_space, len_action, embed_pos, embed_pos.shape[-1], len_feature + embed_pos.shape[-1]
        self.n_head, self.norm = n_head, norm
        self.m, self.n = m, min(n, m)
        self.conscious = True if self.n < self.m else False
        self.dynamics = TRANSFORMER_AUGMENTED(len_object=self.len_object, len_action=self.len_action, num_layers=layers_model, n_head=n_head, FC_depth=FC_depth, FC_width=FC_width, norm=norm)
        self.embed_actions = tf.keras.layers.Embedding(self.n_action_space, self.len_action, embeddings_initializer='identity', trainable=False)
        if self.norm:
            self.downscaler = tf.keras.models.Sequential([
                tf.keras.layers.Conv1D(self.len_feature, kernel_size=1, strides=1),
                tf.keras.layers.LayerNormalization(axis=-1)
            ])
        else:
            self.downscaler = tf.keras.layers.Conv1D(self.len_feature, kernel_size=1, strides=1)
        self.signal_predict_action = bool(signal_predict_action)
        if self.signal_predict_action: # do not use if using given features
            self.len_object_augmented_action_predict = 2 * self.len_object if self.conscious else 2 * self.len_feature + self.len_pos
            self.FC_action_predictor = TRANSFORMER_AUGMENTED(len_object=self.len_object_augmented_action_predict, len_action=0, num_layers=depth_FC_action_predictor, n_head=n_head, FC_depth=1, FC_width=FC_width, norm=norm)
            self.pooler_action_predictor = tf.keras.layers.Dense(n_action_space) # linear and I like it
        self.predictor_reward_term = ESTIMATOR_REWARD_TERM2(len_object=self.len_object, len_action=self.len_action, width_pool=width_pool, depth_transformer=depth_reward_term_predictor, value_min=reward_min, value_max=reward_max, atoms=atoms_reward, norm=norm, n_head=n_head)
        if self.conscious:
            self.compressor = COMPRESSOR_SET(len_object=self.len_object, depth_transformer=1, n_head=self.n_head, size_bottleneck=self.n, len_action=self.len_action, norm=self.norm, FC_width=FC_width, type_compress=type_compress)
            self.decompressor = DECOMPRESSOR_SET(len_object=self.len_object, len_feature=self.len_feature, n_head=self.n_head, len_action=self.len_action, size_bottleneck=self.n)

    @tf.function
    def get_attention(self, obses_curr, actions):
        assert self.conscious
        size_batch = obses_curr.shape[0]
        ebd_actions = self.embed_actions(actions)
        embed_pos = tf.repeat(self.embed_pos, size_batch, axis=0)
        objects_curr = tf.concat([tf.reshape(obses_curr, [size_batch, self.m, -1]), embed_pos], axis=-1)
        _, weights_attention = self.compressor(objects_curr, ebd_actions)
        return weights_attention

    @tf.function
    def __call__(self, features_curr, action, predict_reward=True, predict_term=True, eval=False):
        ebd_action = self.embed_actions(action)
        embed_pos = tf.repeat(self.embed_pos, features_curr.shape[0], axis=0)
        if not eval and self.noisy_shift: embed_pos = noisy_shift2(embed_pos)
        objects_curr = tf.concat([features_curr, embed_pos], axis=-1)
        if self.conscious:
            subset_curr, weights_att_compress = self.compressor(objects_curr, ebd_action)
            subset_imagined = self.rollout_dynamics(subset_curr, ebd_action)
            objects_imagined = self.decompressor(objects_curr, subset_imagined, ebd_action)
            features_imagined = self.downscaler(objects_imagined)
            reward_dist_imagined, term_logits_imagined = self.predictor_reward_term(subset_curr, ebd_action, subset_imagined, predict_reward=predict_reward, predict_term=predict_term)
        else:
            features_imagined, weights_att_compress = self.rollout_dynamics(objects_curr, ebd_action), None
            objects_imagined = tf.concat([features_imagined, embed_pos], axis=-1)
            reward_dist_imagined, term_logits_imagined = self.predictor_reward_term(objects_curr, ebd_action, objects_imagined, predict_reward=predict_reward, predict_term=predict_term)
        return features_imagined, reward_dist_imagined, term_logits_imagined, weights_att_compress

    @tf.function
    def _predict_action(self, features_curr, features_next):
        embed_pos = tf.repeat(self.embed_pos, features_curr.shape[0], axis=0)
        if self.noisy_shift: embed_pos = noisy_shift2(embed_pos)
        objects_augmented = tf.concat([features_curr, features_next, embed_pos], axis=-1)
        objects_augmented = self.FC_action_predictor(objects_augmented)
        summary = tf.reduce_mean(objects_augmented, axis=1)
        logits = self.pooler_action_predictor(summary)
        return logits
    
    @tf.function
    def _predict_action_subset(self, subset_curr, subset_next):
        objects_augmented = tf.concat([subset_curr, subset_next], axis=-1)
        objects_augmented = self.FC_action_predictor(objects_augmented)
        summary = tf.reduce_mean(objects_augmented, axis=1)
        logits = self.pooler_action_predictor(summary)
        return logits

    @tf.function
    def forward_train(self, features_curr, action):
        ebd_action = self.embed_actions(action)
        embed_pos = tf.repeat(self.embed_pos, features_curr.shape[0], axis=0)
        if self.noisy_shift: embed_pos = noisy_shift2(embed_pos)
        objects_curr = tf.concat([features_curr, embed_pos], axis=-1)
        if self.conscious:
            subset_curr, _ = self.compressor(objects_curr, ebd_action)
            subset_imagined = self.rollout_dynamics(subset_curr, ebd_action)
            objects_imagined = self.decompressor(objects_curr, subset_imagined, ebd_action)
            features_imagined = self.downscaler(objects_imagined)
            reward_dist_imagined, term_logits_imagined = self.predictor_reward_term(subset_curr, ebd_action, subset_imagined)
        else:
            features_imagined = self.rollout_dynamics(objects_curr, ebd_action)
            objects_imagined = tf.concat([features_imagined, embed_pos], axis=-1)
            reward_dist_imagined, term_logits_imagined = self.predictor_reward_term(objects_curr, ebd_action, tf.stop_gradient(objects_imagined))
        if self.signal_predict_action:
            if self.conscious:
                action_logits_imagined = self._predict_action_subset(subset_curr, subset_imagined)
            else:
                action_logits_imagined = self._predict_action(features_curr, tf.stop_gradient(features_imagined))
        else:
            action_logits_imagined = None
        return features_imagined, reward_dist_imagined, term_logits_imagined, action_logits_imagined

    @tf.function
    def rollout_dynamics(self, objects, ebd_action):
        objects = self.dynamics(objects, ebd_action)
        if self.conscious:
            return objects
        else:
            features_imagined = self.downscaler(objects)
            return features_imagined

class MODEL_TRANSITION_MINIGRIDOBS(tf.keras.Model): # TODO: implement the observation-level model for Dyna
    def __init__(self, n_action_space, len_action, len_feature, embed_pos, layers_model=3, n_head=8, m=64, n=4, FC_width=64, FC_depth=3, norm=False, depth_reward_term_predictor=1, reward_min=-1, reward_max=1, atoms_reward=64, width_pool=64, type_compress='semihard'):
        # TODO: must add those 0s otherwise the heads are gonna be messed up
        super(MODEL_TRANSITION_MINIGRIDOBS, self).__init__()
        self.len_feature, self.n_action_space, self.len_action, self.embed_pos, self.len_pos, self.len_object = len_feature, n_action_space, len_action, embed_pos, embed_pos.shape[-1], len_feature + embed_pos.shape[-1]
        self.n_head, self.norm = n_head, norm
        self.m, self.n = m, min(n, m)
        self.conscious = True if self.n < self.m else False
        self.dynamics = TRANSFORMER_AUGMENTED(len_object=self.len_object, len_action=self.len_action, num_layers=layers_model, n_head=n_head, FC_depth=FC_depth, FC_width=FC_width, norm=norm)
        self.embed_actions = tf.keras.layers.Embedding(self.n_action_space, self.len_action, embeddings_initializer='identity', trainable=False)
        if self.norm:
            self.downscaler = tf.keras.models.Sequential([
                tf.keras.layers.Conv1D(self.len_feature, kernel_size=1, strides=1),
                tf.keras.layers.LayerNormalization(axis=-1)
            ])
        else:
            self.downscaler = tf.keras.layers.Conv1D(self.len_feature, kernel_size=1, strides=1)
        self.predictor_reward_term = ESTIMATOR_REWARD_TERM2(len_object=self.len_object, len_action=self.len_action, width_pool=width_pool, depth_transformer=depth_reward_term_predictor, value_min=reward_min, value_max=reward_max, atoms=atoms_reward, norm=norm, n_head=n_head)
        if self.conscious:
            self.compressor = COMPRESSOR_SET(len_object=self.len_object, depth_transformer=1, n_head=self.n_head, size_bottleneck=self.n, len_action=self.len_action, norm=self.norm, FC_width=FC_width, type_compress=type_compress)
            self.decompressor = DECOMPRESSOR_SET(len_object=self.len_object, len_feature=self.len_feature, n_head=self.n_head, len_action=self.len_action, size_bottleneck=self.n)
        self.tail_feature = tf.constant(tf.zeros([1, self.m, len_feature - 3], dtype=tf.float32))

    @tf.function
    def __call__(self, obses_curr, actions, predict_reward=True, predict_term=True, eval=False):
        size_batch = obses_curr.shape[0]
        ebd_actions = self.embed_actions(actions)
        embed_pos = tf.repeat(self.embed_pos, size_batch, axis=0)
        tails_feature = tf.repeat(self.tail_feature, size_batch, axis=0)
        objects_curr = tf.concat([tf.reshape(obses_curr, [size_batch, self.m, -1]), tails_feature, embed_pos], axis=-1)
        if self.conscious:
            subset_curr, _ = self.compressor(objects_curr, ebd_actions)
            subset_imagined = self.rollout_dynamics(subset_curr, ebd_actions)
            objects_imagined = self.decompressor(objects_curr, subset_imagined, ebd_actions)
            features_imagined = self.downscaler(objects_imagined)
            reward_dist_imagined, term_logits_imagined = self.predictor_reward_term(subset_curr, ebd_actions, subset_imagined, predict_reward=predict_reward, predict_term=predict_term)
        else:
            features_imagined, _ = self.rollout_dynamics(objects_curr, ebd_actions), None
            objects_imagined = tf.concat([features_imagined, embed_pos], axis=-1)
            reward_dist_imagined, term_logits_imagined = self.predictor_reward_term(objects_curr, ebd_actions, objects_imagined, predict_reward=predict_reward, predict_term=predict_term)
        obses_imagined = tf.reshape(features_imagined[:, :, 0: 3], obses_curr.shape)
        return obses_imagined, reward_dist_imagined, term_logits_imagined

    @tf.function
    def forward_train(self, obses_curr, actions):
        size_batch = obses_curr.shape[0]
        ebd_actions = self.embed_actions(actions)
        embed_pos = tf.repeat(self.embed_pos, size_batch, axis=0)
        tails_feature = tf.repeat(self.tail_feature, size_batch, axis=0)
        objects_curr = tf.concat([tf.reshape(obses_curr, [size_batch, self.m, -1]), tails_feature, embed_pos], axis=-1)
        if self.conscious:
            subset_curr, _ = self.compressor(objects_curr, ebd_actions)
            subset_imagined = self.rollout_dynamics(subset_curr, ebd_actions)
            objects_imagined = self.decompressor(objects_curr, subset_imagined, ebd_actions)
            features_imagined = self.downscaler(objects_imagined)
            reward_dist_imagined, term_logits_imagined = self.predictor_reward_term(subset_curr, ebd_actions, subset_imagined)
        else:
            features_imagined = self.rollout_dynamics(objects_curr, ebd_actions)
            objects_imagined = tf.concat([features_imagined, embed_pos], axis=-1)
            reward_dist_imagined, term_logits_imagined = self.predictor_reward_term(objects_curr, ebd_actions, tf.stop_gradient(objects_imagined))
        obses_imagined = tf.reshape(features_imagined[:, :, 0: 3], obses_curr.shape)
        return obses_imagined, reward_dist_imagined, term_logits_imagined

    @tf.function
    def rollout_dynamics(self, objects, ebd_action):
        objects = self.dynamics(objects, ebd_action)
        if self.conscious:
            return objects
        else:
            features_imagined = self.downscaler(objects)
            return features_imagined

class COMPRESSOR_SET(tf.keras.layers.Layer):
    def __init__(self, len_object=64, depth_transformer=1, n_head=8, size_bottleneck=8, len_action=8, norm=False, FC_depth=3, FC_width=64, type_compress='semihard'):
        super(COMPRESSOR_SET, self).__init__(name='compressor_set')
        self.len_object, self.len_action, self.size_bottleneck = len_object, len_action, size_bottleneck
        if type_compress == 'semihard':
            self.self_attn = MultiHeadAttention(len_object, n_head, top_k=self.size_bottleneck)
        elif type_compress == 'soft':
            self.self_attn = MultiHeadAttention(len_object, n_head)
        else:
            raise NotImplementedError
        self.queries_subset = tf.Variable(tf.keras.initializers.GlorotNormal()([1, self.size_bottleneck, self.len_object]), trainable=True)
        self.layers = TRANSFORMER_AUGMENTED(len_object=self.len_object, len_action=self.len_action, num_layers=depth_transformer, n_head=n_head, FC_depth=FC_depth, FC_width=FC_width, norm=norm)

    @tf.function
    def __call__(self, objects, ebd_action):
        objects = self.layers(objects, ebd_action)
        queries_subset_augmented = tf.repeat(self.queries_subset, objects.shape[0], axis=0) # tf.nn.relu(self.queries_subset)
        subset, weights_attention = self.self_attn(objects, objects, queries_subset_augmented) # V, K, Q
        return subset, weights_attention

class DECOMPRESSOR_SET(tf.keras.layers.Layer): #TODO: to be tested!
    def __init__(self, len_object=64, len_feature=56, n_head=8, len_action=8, size_bottleneck=8, residual=False):
        super(DECOMPRESSOR_SET, self).__init__(name='compressor_set')
        self.len_feature, self.len_object, self.len_action = len_feature, len_object, len_action
        self.residual = residual
        self.self_attn = MultiHeadAttention(len_object, n_head)

    @tf.function
    def __call__(self, objects_in, subset, ebd_action):
        objects_augmented = tf.concat([objects_in, tf.repeat(tf.reshape(ebd_action, [objects_in.shape[0], 1, self.len_action]), objects_in.shape[1], axis=1)], axis=-1) # tf.nn.relu(objects_in)
        objects_tmp, _ = self.self_attn(subset, subset, objects_augmented) # V, K, Q
        return objects_in + objects_tmp if self.residual else objects_tmp

class ESTIMATOR_REWARD_TERM2(tf.keras.layers.Layer):
    def __init__(self, len_object, len_action, width_pool=128, depth_transformer=1, value_min=-1, value_max=1, atoms=128, norm=False, n_head=8):
        super(ESTIMATOR_REWARD_TERM2, self).__init__(name='estimator_reward_term')
        self.value_min, self.value_max, self.atoms = float(value_min), float(value_max), int(atoms)
        self.len_object, self.len_action, self.len_augment = len_object, len_action, 2 * len_object + len_action
        self.mlp = TRANSFORMER_AUGMENTED(len_object=self.len_augment, num_layers=depth_transformer, len_action=0, norm=norm, n_head=n_head)
        self.dim_scaler = tf.keras.layers.Conv1D(width_pool, kernel_size=1, activation='relu', strides=1)
        self.pooler_reward = tf.keras.models.Sequential([
            tf.keras.layers.Dense(width_pool, activation='relu'),
            tf.keras.layers.Dense(width_pool, activation='relu'),
            tf.keras.layers.Dense(self.atoms),
            tf.keras.layers.Softmax(axis=-1),
        ])
        self.pooler_term = tf.keras.models.Sequential([
            tf.keras.layers.Dense(width_pool, activation='relu'),
            tf.keras.layers.Dense(width_pool, activation='relu'),
            tf.keras.layers.Dense(2),
        ])

    @tf.function
    def __call__(self, subset_curr, ebd_action, subset_next, predict_reward=True, predict_term=True):
        if not predict_reward and not predict_term: # save time
            return None, None
        else:
            subset_augmented = tf.concat([subset_curr, subset_next, tf.repeat(tf.expand_dims(ebd_action, 1), subset_curr.shape[1], axis=1)], axis=-1)
            subset_augmented = self.dim_scaler(self.mlp(subset_augmented))
            summary = tf.reduce_mean(subset_augmented, axis=-2)
            reward = self.pooler_reward(summary) if predict_reward else None
            term = self.pooler_term(summary) if predict_term else None
            return reward, term
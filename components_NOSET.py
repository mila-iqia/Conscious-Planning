"""
COMPONENTS DEFINITIONS FOR NOSET baseline
"""

import tensorflow as tf

class OBJECT_EXTRACTOR_NOSET(tf.keras.layers.Layer):
    """noset counterpart for OBJECT_EXTRACTOR"""
    def __init__(self, feature_extractor, norm=False, len_hidden=512):
        super(OBJECT_EXTRACTOR_NOSET, self).__init__(name='extractor_noset')
        self.feature_extractor = feature_extractor
        self.type_env, self.type_extractor = self.feature_extractor.type_env, self.feature_extractor.type_extractor
        self.convh, self.convw, self.m = self.feature_extractor.convh, self.feature_extractor.convw, self.feature_extractor.m
        self.divisor_feature, self.dtype_converted_obs, self.features_learnable = self.feature_extractor.divisor_feature, self.feature_extractor.dtype_converted_obs, self.feature_extractor.features_learnable
        self.len_hidden = len_hidden
        self.scaler = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.len_hidden)
        ])
        self.norm = norm
        if self.norm: self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)

    @tf.function
    def __call__(self, obs):
        x = self.feature_extractor(obs)
        x = self.scaler(x)
        return self.layernorm(x) if self.norm else x

class ESTIMATOR_VALUE_NOSET(tf.keras.layers.Layer):
    """ noset counterpart for ESTIMATOR_VALUE """
    def __init__(self, num_actions, width=64, value_min=-1, value_max=1, atoms=64, transform=False):
        super(ESTIMATOR_VALUE_NOSET, self).__init__(name='head_value')
        self.num_actions = num_actions
        self.value_min, self.value_max, self.atoms, self.transform = float(value_min), float(value_max), int(atoms), bool(transform)
        self.head_Q = tf.keras.models.Sequential([
            tf.keras.layers.Dense(width, activation='relu'),
            tf.keras.layers.Dense(width, activation='relu'),
            tf.keras.layers.Dense(num_actions * self.atoms),
        ])

    @tf.function
    def __call__(self, features, softmax=True, eval=False):
        logits = tf.reshape(self.head_Q(features), (-1, self.num_actions, self.atoms))
        if softmax:
            return tf.nn.softmax(logits, axis=-1)
        else:
            return logits

class ESTIMATOR_REWARD_TERM_NOSET(tf.keras.layers.Layer):
    def __init__(self, width_pool=128, value_min=-1, value_max=1, atoms=128, transform=False):
        super(ESTIMATOR_REWARD_TERM_NOSET, self).__init__(name='estimator_reward_term')
        self.value_min, self.value_max, self.atoms, self.transform = float(value_min), float(value_max), int(atoms), bool(transform) # transform not used in the member methods but will be referred by others!
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
    def __call__(self, feature_curr, ebd_action, feature_next, predict_reward=True, predict_term=True):
        if not predict_reward and not predict_term: # save time
            return None, None
        else:
            feature_augmented = tf.concat([feature_curr, feature_next, ebd_action], axis=-1)
            reward = self.pooler_reward(feature_augmented) if predict_reward else None
            term = self.pooler_term(feature_augmented) if predict_term else None
            return reward, term

class MODEL_TRANSITION_NOSET(tf.keras.Model):
    def __init__(self, n_action_space, len_action, layers_model=3, norm=False, reward_min=-1, reward_max=1, atoms_reward=64, transform_reward=False, signal_predict_action=True, len_hidden=512):
        super(MODEL_TRANSITION_NOSET, self).__init__(name='model_transition')
        self.n_action_space, self.len_action = n_action_space, len_action
        self.norm = norm
        self.dynamics = tf.keras.models.Sequential()
        for layer in range(layers_model):
            if layer < layers_model - 1:
                self.dynamics.add(tf.keras.layers.Dense(len_hidden, activation='relu'))
            else:
                self.dynamics.add(tf.keras.layers.Dense(len_hidden))
        if self.norm: self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)
        self.embed_actions = tf.keras.layers.Embedding(self.n_action_space, self.len_action, embeddings_initializer='identity', trainable=False)
        self.signal_predict_action = bool(signal_predict_action)
        if self.signal_predict_action: self.pooler_action_predictor = tf.keras.layers.Dense(n_action_space) # linear and I like it
        self.predictor_reward_term = ESTIMATOR_REWARD_TERM_NOSET(value_min=reward_min, value_max=reward_max, atoms=atoms_reward, transform=transform_reward)

    @tf.function
    def __call__(self, feature_curr, action, predict_reward=True, predict_term=True, eval=False):
        ebd_action = self.embed_actions(action)
        feature_imagined = self.rollout_dynamics(feature_curr, ebd_action)
        reward_dist_imagined, term_logits_imagined = self.predictor_reward_term(feature_curr, ebd_action, feature_imagined, predict_reward=predict_reward, predict_term=predict_term)
        return feature_imagined, reward_dist_imagined, term_logits_imagined, None

    @tf.function
    def _predict_action(self, feature_curr, feature_next):
        feature_augmented = tf.concat([feature_curr, feature_next], axis=-1)
        logits = self.pooler_action_predictor(feature_augmented)
        return logits

    @tf.function
    def forward_train(self, feature_curr, action):
        ebd_action = self.embed_actions(action)
        feature_imagined = self.rollout_dynamics(feature_curr, ebd_action)
        reward_dist_imagined, term_logits_imagined = self.predictor_reward_term(feature_curr, ebd_action, tf.stop_gradient(feature_imagined))
        if self.signal_predict_action:
            action_logits_imagined = self._predict_action(feature_curr, tf.stop_gradient(feature_imagined))
        else:
            action_logits_imagined = None
        return feature_imagined, reward_dist_imagined, term_logits_imagined, action_logits_imagined

    @tf.function
    def rollout_dynamics(self, feature_curr, ebd_action):
        features_imagined = self.dynamics(tf.concat([feature_curr, ebd_action], axis=-1))
        if self.norm:
            return self.layernorm(feature_curr + features_imagined)
        else:
            return feature_curr + features_imagined
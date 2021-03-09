import tensorflow as tf

config_Conv2D = {'kernel_initializer': 'glorot_normal'}

class RL_AGENT(tf.Module):
    def __init__(self, env, gamma, writer=None, disable_record=False):
        super(RL_AGENT, self).__init__()
        self.initialized = False
        self.env_name = env.spec._env_name
        self.gamma = gamma
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.env = env
        self.writer = writer
        self.disable_record = disable_record

    def record_scalar(self, identifier, value, step):
        if self.writer is not None:
            self.writer.add_scalar(identifier, float(value), step)
        else:
            try:
                tf.summary.scalar(identifier, value, step)
            except:
                pass

    def record_image(self, identifier, image, step):
        if self.writer is not None:
            self.writer.add_image(identifier, image, step)
        else:
            try:
                tf.summary.scalar(identifier, image, step)
            except:
                pass

class BLOCK_RESIDUAL(tf.keras.layers.Layer): # impala has width unit 32, if scaled by 4 it should be 128
    def __init__(self, len_feature, width=None, kernel_size=(3, 3), strides=(1, 1)):
        super(BLOCK_RESIDUAL, self).__init__(name='block_residual')
        if width is None: width = len_feature
        self.bypass = tf.keras.models.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(width, kernel_size=kernel_size, strides=strides, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(len_feature, kernel_size=kernel_size, strides=strides, padding='same'),
        ])

    @tf.function
    def __call__(self, input_tensor):
        return input_tensor + self.bypass(input_tensor)

class BLOCK_IMPALA(tf.keras.layers.Layer): # impala has width unit 32, if scaled by 4 it should be 128
    def __init__(self, scale=4):
        super(BLOCK_IMPALA, self).__init__(name='block_impala')
        self.conv_pool = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(int(scale * 16), kernel_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        ])
        self.block_first = BLOCK_RESIDUAL(int(scale * 16), width=int(scale * 32))
        self.block_second = BLOCK_RESIDUAL(int(scale * 16), width=int(scale * 32))

    @tf.function
    def __call__(self, x):
        x = self.conv_pool(x)
        x = self.block_first(x)
        x = self.block_second(x)
        return x

class EXTRACTOR_FEATURE_PROCGEN(tf.keras.layers.Layer):
    def __init__(self, shape_input, len_output=32, learnable=True, num_blocks=3, scale=4):
        super(EXTRACTOR_FEATURE_PROCGEN, self).__init__()
        self.h, self.w, self.channels_in = shape_input[-3], shape_input[-2], shape_input[-1]
        self.convh, self.convw = self.h, self.w
        self.len_output = len_output
        self.learnable = learnable
        self.blocks = []
        for _ in range(num_blocks): self.blocks.append(BLOCK_IMPALA(scale=scale))
        if len_output == int(scale * 16):
            self.scaler = None
        else:
            self.scaler = tf.keras.models.Sequential([
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(len_output, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            ])
        if not self.learnable:
            for block in self.blocks: block.trainable = False
            self.scaler.trainable = False
    
    @tf.function
    def __call__(self, x):
        for block in self.blocks: x = block(x)
        if self.scaler is not None: x = self.scaler(x)
        return x if self.learnable else tf.stop_gradient(x)

class EXTRACTOR_FEATURE_MUZERO(tf.keras.layers.Layer): # From Ji
    def __init__(self, shape_input, len_output=64):
        super(EXTRACTOR_FEATURE_MUZERO, self).__init__(name='layers_extractor_feature_muzero')
        self.h, self.w, self.channels_in = shape_input[-3], shape_input[-2], shape_input[-1]
        self.convh, self.convw = 13, 10
        self.len_output = len_output
        kernel_size = (3, 3)
        self.conv1 = tf.keras.layers.Conv2D(64, input_shape=(self.h, self.w, self.channels_in), kernel_size=kernel_size, strides=2, padding='same')
        self.resi2 = BLOCK_RESIDUAL(64)
        self.resi3 = BLOCK_RESIDUAL(64)
        self.conv4 = tf.keras.layers.Conv2D(128, kernel_size=kernel_size, strides=2, padding='same')
        self.resi5 = BLOCK_RESIDUAL(128)
        self.resi9 = BLOCK_RESIDUAL(128)
        self.pool12 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)
        if self.len_output != 128:
            self.trimmer = tf.keras.layers.Conv2D(self.len_output, kernel_size=(1, 1), padding='same')
        else:
            self.trimmer = None
        
            
    @tf.function
    def __call__(self, x):
        x = self.conv1(x)
        x = self.resi2(x)
        x = self.resi3(x)
        x = self.conv4(x)
        x = self.resi5(x)
        x = self.resi9(x)
        x = self.pool12(x)
        if self.trimmer is not None: x = self.trimmer(x)
        return x

class EXTRACTOR_FEATURE_MINIGRID(tf.keras.layers.Layer):
    def __init__(self, shape_input, len_output=64):
        super(EXTRACTOR_FEATURE_MINIGRID, self).__init__(name='layers_extractor_feature_minigrid')
        self.h, self.w, self.channels_in = shape_input[-3], shape_input[-2], shape_input[-1]
        self.convh, self.convw = self.h, self.w
        self.len_output = len_output
        self.tails = tf.zeros([1, self.h, self.w, len_output - self.channels_in])

    @tf.function
    def __call__(self, x):
        size_batch = x.shape[0]
        x = tf.concat([x, tf.repeat(self.tails, size_batch, axis=0)], axis=-1)
        return tf.stop_gradient(x)

class EXTRACTOR_FEATURE_MINIGRID_BOW(tf.keras.layers.Layer):
    def __init__(self, shape_input, len_output=32, value_max=31, learnable=True):
        super(EXTRACTOR_FEATURE_MINIGRID_BOW, self).__init__()
        self.h, self.w, self.channels_in = shape_input[-3], shape_input[-2], shape_input[-1]
        self.convh, self.convw = self.h, self.w
        self.len_output = len_output
        self.learnable = learnable
        embedding_matrix = tf.random.normal([3 * (value_max + 1), len_output], mean=0.0, stddev=1.0)
        embedding_matrix = embedding_matrix / tf.norm(embedding_matrix, ord='euclidean', axis=-1, keepdims=True)
        self.embedder = tf.keras.layers.Embedding(3 * (value_max + 1), len_output, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix))
        self.offsets = tf.cast(tf.reshape(tf.constant([0, value_max + 1, 2 * (value_max + 1)]), [1, 1, 1, 3]), tf.int32)

    @tf.function
    def __call__(self, x):
        embeddings_per_dim = self.embedder(x + self.offsets)
        embeddings = tf.reduce_sum(embeddings_per_dim, axis=-2)
        if self.learnable:
            return embeddings
        else:
            return tf.stop_gradient(embeddings)

class EXTRACTOR_FEATURE_MINIGRID_MLP(tf.keras.layers.Layer):
    def __init__(self, shape_input, len_output=64, depth=1, width=64, learnable=True):
        super(EXTRACTOR_FEATURE_MINIGRID_MLP, self).__init__()
        self.h, self.w, self.channels_in = shape_input[-3], shape_input[-2], shape_input[-1]
        self.convh, self.convw = self.h, self.w
        self.len_output = len_output
        self.learnable = learnable
        if depth == 1:
            self.encoder = tf.keras.layers.Conv2D(self.len_output, kernel_size=1, strides=1, **config_Conv2D)
        else:
            self.encoder = tf.keras.models.Sequential()
            for num_layer in range(depth):
                if num_layer < depth - 1:
                    self.encoder.add(tf.keras.layers.Conv2D(width, kernel_size=1, strides=1, activation='relu', **config_Conv2D))
                else:
                    self.encoder.add(tf.keras.layers.Conv2D(self.len_output, kernel_size=1, strides=1, **config_Conv2D))

    @tf.function
    def __call__(self, obs):
        if self.learnable:
            return self.encoder(obs)
        else:
            return tf.stop_gradient(self.encoder(obs))

# class EXTRACTOR_FEATURE_MUZERO(tf.keras.layers.Layer):
#     def __init__(self, shape_input, len_output=64):
#         super(EXTRACTOR_FEATURE_MUZERO, self).__init__(name='layers_extractor_feature_muzero')
#         self.h, self.w, self.channels_in = shape_input[-3], shape_input[-2], shape_input[-1]
#         self.convh, self.convw = 27, 20
#         self.len_output = len_output
#         kernel_size, stride = (3, 3), (2, 2)
#         self.conv1 = tf.keras.layers.Conv2D(64, input_shape=(self.h, self.w, self.channels_in), kernel_size=kernel_size, strides=stride, padding='same') # , activation='relu'
#         self.resi2 = BLOCK_RESIDUAL(64)
#         self.resi3 = BLOCK_RESIDUAL(64)
#         self.conv4 = tf.keras.layers.Conv2D(256, kernel_size=kernel_size, strides=stride, padding='same')
#         self.resi5 = BLOCK_RESIDUAL(256)
#         self.resi6 = BLOCK_RESIDUAL(256)
#         if self.len_output != 256:
#             self.trimmer = tf.keras.layers.Conv2D(self.len_output, kernel_size=(1, 1), padding='same')
#         else:
#             self.trimmer = None
            
#     @tf.function
#     def __call__(self, x):
#         x = self.conv1(x)
#         x = self.resi2(x)
#         x = self.resi3(x)
#         x = self.conv4(x)
#         x = self.resi5(x)
#         x = self.resi6(x)
#         if self.trimmer is not None: x = self.trimmer(x)
#         return x

class EXTRACTOR_FEATURE(tf.keras.layers.Layer):
    def __init__(self, shape_input, channels_out=64, type_extractor='minigrid', features_learnable=True):
        super(EXTRACTOR_FEATURE, self).__init__()
        self.type_extractor = type_extractor
        self.channels_out = channels_out
        self.h, self.w, self.channels_in = shape_input[-3], shape_input[-2], shape_input[-1]
        if 'minigrid' in type_extractor:
            self.type_env = 'minigrid'
            self.convw, self.convh = self.w, self.h
            if 'bow' in type_extractor:
                self.divisor_feature, self.dtype_converted_obs, self.features_learnable = None, tf.int32, True and features_learnable
                self.extractor = EXTRACTOR_FEATURE_MINIGRID_BOW(shape_input=[self.h, self.w, self.channels_in], len_output=self.channels_out, learnable=self.features_learnable)
            else:
                self.divisor_feature, self.dtype_converted_obs, self.features_learnable = None, tf.float32, False and features_learnable
                self.extractor = EXTRACTOR_FEATURE_MINIGRID(shape_input=[self.h, self.w, self.channels_in], len_output=self.channels_out)
        else:
            raise NotImplementedError
        self.m = self.convh * self.convw

    @tf.function
    def __call__(self, x):
        return self.extractor(x)
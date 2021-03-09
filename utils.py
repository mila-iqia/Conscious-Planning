from runtime import *
import tensorflow as tf
import numpy as np

# EPSILON = np.finfo(tf.float32.as_numpy_dtype).tiny

# @tf.function
# def gumbel_keys(w): # sample some gumbels, adding gumbel perturbation to the weights
#     return w + tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(w), minval=EPSILON, maxval=1.0)))

# @tf.function
# def continuous_topk(w, k, t, separate=False):
#     khot_list = []
#     onehot_approx = tf.zeros_like(w, dtype=tf.float32)
#     for _ in range(k):
#         khot_mask = tf.maximum(1.0 - onehot_approx, EPSILON)
#         w += tf.math.log(khot_mask) # accummulating log-softmax
#         onehot_approx = tf.nn.softmax(w / t, axis=-1)
#         khot_list.append(onehot_approx)
#     if separate:
#         return khot_list
#     else:
#         return tf.reduce_sum(khot_list, 0)

# @tf.function
# def sample_subset(w, k, t=0.1):
#     '''
#     w (Tensor): Float Tensor of weights for each element. In gumbel mode these are interpreted as log probabilities
#     k (int): number of elements in the subset sample
#     t (float): temperature of the softmax
#     '''
#     return continuous_topk(gumbel_keys(w), k, t)

@tf.function
def to_categorical(value, value_min=-1, value_max=1, atoms=128, clip=True):
    if clip: value = tf.clip_by_value(value, clip_value_min=value_min, clip_value_max=value_max)
    value = (value - value_min) * (atoms - 1) / (value_max - value_min)
    upper = tf.cast(tf.math.ceil(value), dtype=tf.int32)
    upper_weight = value % 1
    lower = tf.cast(tf.math.floor(value), dtype=tf.int32)
    lower_weight = 1 - upper_weight
    span = tf.range(value.shape[0], dtype=tf.int32)
    indices_upper = tf.stack([span, upper], axis=-1)
    indices_lower = tf.stack([span, lower], axis=-1)
    dist = tf.scatter_nd(indices_upper, upper_weight, value.shape + [atoms])
    dist = tf.tensor_scatter_nd_update(dist, indices=indices_lower, updates=lower_weight)
    return dist

@tf.function
def from_categorical(dist, value_min=-1, value_max=1, atoms=128):
    support = tf.expand_dims(tf.cast(tf.range(start=0, limit=atoms, delta=1), dtype=tf.float32), axis=-1)
    value = tf.squeeze(dist @ support, [-1])
    value = value_min + value * (value_max - value_min) / (atoms - 1)
    return value

def embed_pos_hd(dims, len_embed_pos=8):
    dims = list(dims)
    convh, convw = dims[-2], dims[-1]
    embed_pos1 = np.zeros((convh, convw, 2))
    for i in range(convh):
        for j in range(convw):
            embed_pos1[i, j, 0] = i
            embed_pos1[i, j, 1] = j 
    embed_pos1 = np.reshape(embed_pos1, (-1, 2))

    embed_pos2 = np.zeros((convh, convw, 2))
    for i in range(convh):
        for j in range(convw):
            embed_pos2[i, j, 0] = convh - i - 1
            embed_pos2[i, j, 1] = j
    embed_pos2 = np.reshape(embed_pos2, (-1, 2))

    embed_pos3 = np.zeros((convh, convw, 2))
    for i in range(convh):
        for j in range(convw):
            embed_pos3[i, j, 0] = i
            embed_pos3[i, j, 1] = convw - j - 1
    embed_pos3 = np.reshape(embed_pos3, (-1, 2))

    embed_pos4 = np.zeros((convh, convw, 2))
    for i in range(convh):
        for j in range(convw):
            embed_pos4[i, j, 0] = convh - i - 1
            embed_pos4[i, j, 1] = convw - j - 1
    embed_pos4 = np.reshape(embed_pos4, (-1, 2))
    embed_pos = np.stack([embed_pos1[:, 0], embed_pos2[:, 0], embed_pos3[:, 0], embed_pos4[:, 0], embed_pos1[:, 1], embed_pos2[:, 1], embed_pos3[:, 1], embed_pos4[:, 1]], axis=-1)
    # embed_pos = np.concatenate([embed_pos1, embed_pos2, embed_pos3, embed_pos4], axis=-1)
    dim_optimal = 8
    embed_pos = tf.convert_to_tensor(embed_pos, dtype=tf.float32)
    embed_pos = tf.expand_dims(embed_pos, 0)
    if len_embed_pos < 8:
        assert len_embed_pos % 2 == 0
        embed_pos = embed_pos[:, :, 0: len_embed_pos]
        dim_optimal = len_embed_pos
    return embed_pos, dim_optimal

def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)
    if x.get_shape().dims is None: # If unknown rank, return dynamic shape
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

@tf.function
def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """ Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different frequency and phase in one of the positional dimensions.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the memory inputs to attention.
    The use of relative position is possible because sin(a+b) and cos(a+b) can be experessed in terms of b, sin(a) and cos(a).
    x is a Tensor with n "positional" dimensions, e.g. one dimension for a sequence or two dimensions for an image
    We use a geometric sequence of timescales starting with min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we generate the two sinusoidal signals sin(timestep/timescale) and cos(timestep/timescale).  All of these sinusoids are concatenated in the channels dimension.
    Args:       x: a Tensor with shape [batch, d1 ... dn, channels]
                min_timescale: a float; max_timescale: a float
    Returns:    a Tensor the same shape as x. """
    num_dims = len(x.get_shape().as_list()) - 2
    channels = shape_list(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (tf.math.log(float(max_timescale) / float(min_timescale)) / (tf.cast(num_timescales, dtype=tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(tf.range(num_timescales, dtype=tf.float32) * -log_timescale_increment)
    for dim in range(num_dims):
        length = shape_list(x)[dim + 1]
        position = tf.range(length, dtype=tf.float32)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in range(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in range(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal
    return x

@tf.function
def huber_from_L1(abs_error, delta=1.0):
	quadratic = tf.math.minimum(abs_error, delta)
	linear = abs_error - quadratic
	return tf.reduce_mean(0.5 * tf.math.multiply(quadratic, quadratic) + delta * linear, axis=-1)

@tf.function
def noisy_shift2(embed_pos):
    size_batch = embed_pos.shape[0]
    embed_posx, embed_posy = tf.split(embed_pos[0, :, :], 2, axis=-1)
    x_max, y_max = tf.cast(tf.reduce_max(embed_posx), dtype=tf.int32), tf.cast(tf.reduce_max(embed_posy), dtype=tf.int32)
    noise_x = tf.random.uniform([size_batch, 1, embed_posx.shape[-1]], minval=-x_max, maxval=x_max, dtype=tf.int32) # num_objects axis to be broadcasted
    noise_y = tf.random.uniform([size_batch, 1, embed_posy.shape[-1]], minval=-y_max, maxval=y_max, dtype=tf.int32) # num_objects axis to be broadcasted
    noise = tf.cast(tf.concat([noise_x, noise_y], axis=-1), dtype=tf.float32)
    return embed_pos + noise

@tf.function
def clip_gradients(gradients):
    return [None if grad is None else tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1) for grad in gradients]

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, len_object, num_heads, QKV_depth=1, QKV_width=256, top_k=np.inf):
        super(MultiHeadAttention, self).__init__(name='MHA')
        self.top_k = top_k
        self.len_object, self.num_heads = len_object, num_heads
        if num_heads == 0: self.num_heads = len_object
        assert len_object % self.num_heads == 0
        self.depth = len_object // self.num_heads
        if QKV_depth == 1:
            self.wq, self.wk, self.wv = tf.keras.layers.Dense(len_object), tf.keras.layers.Dense(len_object), tf.keras.layers.Dense(len_object)
        else:
            self.wq, self.wk, self.wv = tf.keras.models.Sequential(), tf.keras.models.Sequential(), tf.keras.models.Sequential()
            for num_layer in range(QKV_depth):
                if num_layer == QKV_depth - 1: # last layer
                    self.wq.add(tf.keras.layers.Dense(len_object))
                    self.wk.add(tf.keras.layers.Dense(len_object))
                    self.wv.add(tf.keras.layers.Dense(len_object))
                else:
                    self.wq.add(tf.keras.layers.Dense(QKV_width, activation='relu'))
                    self.wk.add(tf.keras.layers.Dense(QKV_width, activation='relu'))
                    self.wv.add(tf.keras.layers.Dense(QKV_width, activation='relu'))
        self.dense = tf.keras.layers.Dense(len_object)

    @tf.function
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @tf.function
    def __call__(self, v, k, q):
        batch_size = q.shape[0]
        q = self.split_heads(self.wq(q), batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(self.wk(k), batch_size) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(self.wv(v), batch_size) # (batch_size, num_heads, seq_len_v, depth)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, top_k=self.top_k) # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth), attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.len_object))  # (batch_size, seq_len_q, len_object)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, len_object)
        return output, attention_weights
    
@tf.function
def scaled_dot_product_attention(q, k, v, top_k=np.inf):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
    Returns:
        output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(k.shape[-1], tf.float32)) # dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    size_batch, num_heads, num_queries, num_keys = attention_weights.shape
    if top_k < num_keys:
        attention_weights_top_k, indices_top_k = tf.math.top_k(attention_weights, k=top_k, sorted=True)
        indices_stacked = tf.concat([tf.repeat(tf.expand_dims(tf.concat([tf.repeat(tf.expand_dims(tf.stack([tf.repeat(tf.expand_dims(tf.range(size_batch, dtype=tf.int32), axis=-1), num_heads, axis=1), tf.repeat(tf.reshape(tf.range(num_heads, dtype=tf.int32), [1, num_heads]), size_batch, axis=0)], -1), 2), num_queries, axis=2), tf.reshape(tf.range(num_queries, dtype=tf.int32), [1, 1, num_queries, 1]) + tf.zeros([size_batch, num_heads, num_queries, 1], dtype=tf.int32)], axis=-1), axis=-2), top_k, axis=-2), tf.expand_dims(indices_top_k, -1)], axis=-1)
        attention_weights = tf.scatter_nd(tf.stop_gradient(indices_stacked), attention_weights_top_k, [size_batch, num_heads, num_queries, num_keys])
        attention_weights, _ = tf.linalg.normalize(attention_weights, ord=1, axis=-1)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

@tf.function
def mask_change_minigrid(obs_t, obs_tp1):
    size_batch, _, _, len_feature = obs_t.shape
    obs_t = tf.reshape(obs_t, [size_batch, -1, len_feature])
    obs_tp1 = tf.reshape(obs_tp1, [size_batch, -1, len_feature])
    mask_cheat = tf.math.reduce_any(tf.not_equal(obs_t, obs_tp1), axis=-1)
    return mask_cheat

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
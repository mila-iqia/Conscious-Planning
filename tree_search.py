from utils import from_categorical, mask_change_minigrid
import numpy as np, tensorflow as tf
from anytree import NodeMixin
import bisect, copy

class NODE(NodeMixin):
    def __init__(self, U_imag, action_backtrack=None, parent=None, term_imag=False, root=False):
        super(NODE, self).__init__()
        self.U_imag = U_imag
        self.in_cum_reward_imag = 0
        self.term_imag = term_imag
        self.parent = parent
        self._is_root = root
        self.action_backtrack = action_backtrack # the action for depth 1 for instant backtracking

@tf.function
def _get_q_max_from_U(U_imag_nodes_branchable, Q):
    qs_vec_piece = from_categorical(Q(U_imag_nodes_branchable, eval=True), value_min=Q.value_min, value_max=Q.value_max, atoms=Q.atoms, transform=Q.transform)
    return tf.math.reduce_max(qs_vec_piece)

@tf.function
def _get_q_from_U(U_imag_nodes_branchable, Q):
    qs_vec_piece = from_categorical(Q(U_imag_nodes_branchable, eval=True), value_min=Q.value_min, value_max=Q.value_max, atoms=Q.atoms, transform=Q.transform)
    return tf.squeeze(qs_vec_piece)

@tf.function
def _get_predictions(U_imag_nodes_origin, action_chosen, M):
    Us_imag, Rs_next_imag, term_next_imag, weights_attention = M(U_imag_nodes_origin, action_chosen, eval=True) # tf.expand_dims(action_chosen, 0)
    term_next_imag = tf.math.argmax(term_next_imag, axis=-1, output_type=tf.int32)
    Rs_next_imag = from_categorical(Rs_next_imag, value_min=M.predictor_reward_term.value_min, value_max=M.predictor_reward_term.value_max, atoms=M.predictor_reward_term.atoms, transform=M.predictor_reward_term.transform)
    return Us_imag, Rs_next_imag, term_next_imag, weights_attention

@tf.function
def _get_highlight(U_imag_nodes_origin, action_chosen, mask_change, M):
    weights_attention = M.get_attention(U_imag_nodes_origin, action_chosen)
    weights_attention_binary = tf.cast(tf.cast(weights_attention, tf.bool), tf.float32)
    mask_by_head = tf.clip_by_value(tf.reduce_sum(weights_attention_binary, 2), 0, 1)
    covered_by_head = mask_by_head * tf.expand_dims(tf.cast(mask_change, tf.float32), 1)
    num_covered_by_head = tf.squeeze(tf.reduce_sum(covered_by_head, -1))
    idx_head_best = tf.argmax(num_covered_by_head)
    num_covered_by_best_head = tf.gather(num_covered_by_head, idx_head_best)
    if num_covered_by_best_head > 0:
        mask_highlight = tf.cast(tf.gather(tf.squeeze(mask_by_head, 0), idx_head_best), tf.bool)
    else:
        mask_highlight = tf.zeros([M.m], tf.bool)
    return mask_highlight, num_covered_by_best_head

@tf.function
def _get_acc_compress(weights_attention, mask_change):
    mask_change_float = tf.cast(mask_change, tf.float32)
    num_changes = tf.reduce_sum(mask_change_float)
    if num_changes > 0:
        weights_attention_binary = tf.cast(tf.cast(weights_attention, tf.bool), tf.float32)
        acc_overall = tf.reduce_max(tf.reduce_sum(tf.clip_by_value(tf.reduce_sum(weights_attention_binary * tf.reshape(mask_change_float, [1, 1, 1, -1]), 1), 0, 1), -1)) / num_changes
        acc_average = tf.reduce_max(tf.reduce_sum(tf.reduce_mean(weights_attention_binary * tf.reshape(mask_change_float, [1, 1, 1, -1]), 1), -1)) / num_changes
        return acc_overall, acc_average
    else:
        return tf.constant(-1.0, tf.float32), tf.constant(-1.0, tf.float32)

def best_first_search(obs_root, set_actions, M, Q, gamma=0.99, max_rollouts=100, env_root=None, flag_record=False, E=None, t=None, bow=False, flag_eval=False, suffix_record='', func_record_scalar=tf.summary.scalar, func_record_image=tf.summary.image):
    """
    M for model
    Q for q = Q(U)
    E for U = encoder(obs)
    """
    POOL_nodes, POOL_actions, POOL_values, POOL_nodes_leaf, POOL_values_leaf = [], [], [], [], []
    action_intuitive, rollouts, height = None, 0, 0
    node_root = node_unbranched = NODE(E(obs_root), root=True)
    if flag_record:
        prefix = 'Plan_Eval' + suffix_record if flag_eval else 'Plan' + suffix_record
        if env_root is not None:
            node_unbranched.env_clone = copy.deepcopy(env_root)
            node_unbranched.term_true, node_unbranched.in_cum_reward_true, node_unbranched.obs = False, 0.0, np.expand_dims(obs_root, 0)
            node_unbranched.mask = tf.zeros([1, node_unbranched.U_imag.shape[1]], dtype=tf.bool)
            node_unbranched.U_true = node_unbranched.U_imag
            list_acc_picks_overall, list_acc_picks_average = [], []
    while True:
        if node_unbranched.term_imag:
            pos = bisect.bisect_left(POOL_values_leaf, node_unbranched.in_cum_reward_imag); POOL_values_leaf.insert(pos, node_unbranched.in_cum_reward_imag); POOL_nodes_leaf.insert(pos, node_unbranched)
        else: # non-term, the root cannot be terminal
            values_before_imagine_branchable = node_unbranched.in_cum_reward_imag + (gamma ** node_unbranched.depth * _get_q_from_U(node_unbranched.U_imag, Q)).numpy() # R + \gamma R + ... + \gamma ^ n Q
            for index_action in set_actions:
                value = values_before_imagine_branchable[index_action]
                pos = bisect.bisect_left(POOL_values, value); POOL_nodes.insert(pos, node_unbranched); POOL_actions.insert(pos, index_action); POOL_values.insert(pos, value)  
        if len(POOL_values) == 0: break # tree depleted
        node_origin, action_chosen, value_branch_estimated = POOL_nodes.pop(), POOL_actions.pop(), POOL_values.pop()
        if action_intuitive is None: action_intuitive = action_chosen # best action estimated with Q before planning
        if rollouts >= max_rollouts: break
        U_imag, R_next_imag, term_next_imag, weights_attention = _get_predictions(node_origin.U_imag, tf.constant([action_chosen], dtype=tf.int32), M)
        action_backtrack = node_origin.action_backtrack if node_origin.depth else action_chosen # for tracing the action faster
        discount_prod = gamma ** node_origin.depth
        node_unbranched = NODE(U_imag=U_imag, parent=node_origin, action_backtrack=action_backtrack) # , action_in=action_chosen
        node_unbranched.term_imag, node_unbranched.in_cum_reward_imag = bool(term_next_imag), node_origin.in_cum_reward_imag + float(R_next_imag) * discount_prod
        if flag_record and env_root is not None:
            node_unbranched.env_clone = copy.deepcopy(node_origin.env_clone)
            obs_next, R_next_true, node_unbranched.term_true, _ = node_unbranched.env_clone.step(action_chosen)
            node_unbranched.in_cum_reward_true, node_unbranched.obs = node_origin.in_cum_reward_true + float(not node_origin.term_true) * R_next_true * discount_prod, np.expand_dims(obs_next, 0)
            mask_change = mask_change_minigrid(node_origin.obs, node_unbranched.obs)
            if M.conscious:
                acc_picks_overall, acc_picks_average = _get_acc_compress(weights_attention, mask_change)
                acc_picks_overall, acc_picks_average = acc_picks_overall.numpy(), acc_picks_average.numpy()
                if acc_picks_overall > 0:
                    list_acc_picks_overall.append(acc_picks_overall)
                    list_acc_picks_average.append(acc_picks_average)
                if func_record_image is not None and R_next_true > 0 and node_unbranched.term_true:
                    env_clone = copy.deepcopy(node_origin.env_clone)
                    mask_highlight, flag_highlight = _get_highlight(node_origin.U_imag, tf.constant([action_chosen]), tf.constant(mask_change), M)
                    func_record_image('%s/vis_attention_%d' % (prefix, action_chosen), env_clone.attention_render(highlight_mask=mask_highlight.numpy()), t)
            node_unbranched.mask = tf.logical_or(node_origin.mask, mask_change)
        if node_unbranched.depth > height: height = node_unbranched.depth
        rollouts += 1
    # tree search finished, find the node with best value
    if flag_record:
        if len(POOL_nodes_leaf) and (POOL_values_leaf[-1] > value_branch_estimated or len(POOL_values) == 0):
            node_best, value_selected = POOL_nodes_leaf[-1], POOL_values_leaf[-1]
        else:
            node_best, value_selected = node_origin, value_branch_estimated
    else:
        if len(POOL_nodes_leaf) and (POOL_values_leaf[-1] > value_branch_estimated or len(POOL_values) == 0):
            node_best = POOL_nodes_leaf[-1]
        else:
            node_best = node_origin
    # backtracking action to root, find the best action
    action_best = action_chosen if node_best._is_root else node_best.action_backtrack
    if flag_record:
        func_record_scalar('%s/depth_max' % (prefix), height, t)
        func_record_scalar('%s/agreement' % (prefix), action_intuitive == action_best, t)
        func_record_scalar('%s/depth_stop_selected' % (prefix), node_best.depth, t)
        if env_root is not None:
            if not node_best._is_root:
                node_best.U_true = E(node_best.obs)
                diff_in_cum_reward = np.abs(node_best.in_cum_reward_true - node_best.in_cum_reward_imag)
                func_record_scalar('%s/diff_reward_cum_selected' % (prefix), diff_in_cum_reward, t)
                diff_U = tf.abs(node_best.U_true - node_best.U_imag)
                diff_U_significant = diff_U if bow else diff_U[:, :, 0: 3]
                diff_U_elementwise = tf.reduce_mean(diff_U_significant)
                diff_U_elementwise_changed_objects = tf.reduce_mean(tf.boolean_mask(diff_U_significant, node_best.mask))
                diff_U_elementwise_unchanged_objects = tf.reduce_mean(tf.boolean_mask(diff_U_significant, tf.logical_not(node_best.mask)))
                func_record_scalar('%s/diff_feature_elementwise_selected' % (prefix), diff_U_elementwise, t)
                if not tf.math.is_nan(diff_U_elementwise_changed_objects): func_record_scalar('%s/diff_U_elementwise_changed_objects' % (prefix), diff_U_elementwise_changed_objects, t)
                if not tf.math.is_nan(diff_U_elementwise_unchanged_objects): func_record_scalar('%s/diff_U_elementwise_unchanged_objects' % (prefix), diff_U_elementwise_unchanged_objects, t)
            if node_best.term_imag:
                diff_value = tf.abs(node_best.in_cum_reward_true - value_selected)
            else:
                Q_max_node_best = _get_q_max_from_U(node_best.U_true, Q)
                diff_value = tf.abs(node_best.in_cum_reward_true + Q_max_node_best - value_selected)
            func_record_scalar('%s/diff_value_selected' % (prefix), diff_value, t)
            if M.conscious and len(list_acc_picks_overall):
                func_record_scalar('%s/acc_picks_overall' % (prefix), np.mean(list_acc_picks_overall), t)
                func_record_scalar('%s/acc_picks_average' % (prefix), np.mean(list_acc_picks_average), t)
                if func_record_image is not None:
                    env_clone = copy.deepcopy(node_root.env_clone)
                    obs_next = np.expand_dims(env_clone.step(action_best)[0], 0)
                    mask_change = mask_change_minigrid(node_root.obs, obs_next)
                    mask_highlight, flag_highlight = _get_highlight(node_root.U_imag, tf.constant([action_best]), tf.constant(mask_change), M)
                    if flag_highlight.numpy():
                        func_record_image('%s/vis_attention_%d' % (prefix, action_best), env_root.attention_render(highlight_mask=mask_highlight.numpy()), t)
    return action_best

def random_search(obs_root, set_actions, M, Q, gamma=0.99, max_rollouts=100, env_root=None, flag_record=False, E=None, t=None, bow=False, flag_eval=False, suffix_record='', func_record_scalar=tf.summary.scalar, func_record_image=tf.summary.image):
    POOL_nodes, POOL_actions, POOL_values, POOL_priorities, POOL_nodes_leaf, POOL_values_leaf = [], [], [], [], [], []
    action_intuitive, rollouts, height = None, 0, 0
    node_root = node_unbranched = NODE(E(obs_root), root=True)
    if flag_record:
        prefix = 'Plan_Eval' + suffix_record if flag_eval else 'Plan' + suffix_record
        if env_root is not None:
            node_unbranched.env_clone = copy.deepcopy(env_root)
            node_unbranched.term_true, node_unbranched.in_cum_reward_true, node_unbranched.obs = False, 0.0, np.expand_dims(obs_root, 0)
            node_unbranched.mask = tf.zeros([1, node_unbranched.U_imag.shape[1]], dtype=tf.bool)
            node_unbranched.U_true = node_unbranched.U_imag
            list_acc_picks_overall, list_acc_picks_average = [], []
    while True:
        if node_unbranched.term_imag:
            pos = bisect.bisect_left(POOL_values_leaf, node_unbranched.in_cum_reward_imag); POOL_values_leaf.insert(pos, node_unbranched.in_cum_reward_imag); POOL_nodes_leaf.insert(pos, node_unbranched)
        else: # non-term, the root cannot be terminal
            values_before_imagine_branchable = node_unbranched.in_cum_reward_imag + (gamma ** node_unbranched.depth * _get_q_from_U(node_unbranched.U_imag, Q)).numpy() # R + \gamma R + ... + \gamma ^ n Q
            for index_action in set_actions:
                value = float(values_before_imagine_branchable[index_action])
                priority = np.random.rand()
                pos = bisect.bisect_left(POOL_priorities, priority)
                POOL_nodes.insert(pos, node_unbranched); POOL_actions.insert(pos, index_action); POOL_values.insert(pos, value); POOL_priorities.insert(pos, priority)
        if len(POOL_priorities) == 0: break
        index_current_best_nonleaf = np.argmax(POOL_values)
        value_current_best_nonleaf = POOL_values[index_current_best_nonleaf]
        node_current_best_nonleaf = POOL_nodes[index_current_best_nonleaf]
        action_current_best_nonleaf = POOL_actions[index_current_best_nonleaf]
        node_origin, action_chosen, _, _ = POOL_nodes.pop(), POOL_actions.pop(), POOL_values.pop(), POOL_priorities.pop() # sorted with the priorities
        if action_intuitive is None: action_intuitive = action_current_best_nonleaf # best action estimated with Q before planning
        if rollouts >= max_rollouts: break
        U_imag, R_next_imag, term_next_imag, weights_attention = _get_predictions(node_origin.U_imag, tf.constant([action_chosen], dtype=tf.int32), M)
        action_backtrack = node_origin.action_backtrack if node_origin.depth else action_chosen # for tracing the action faster
        discount_prod = gamma ** node_origin.depth
        node_unbranched = NODE(U_imag=U_imag, parent=node_origin, action_backtrack=action_backtrack) # , action_in=action_chosen
        node_unbranched.term_imag, node_unbranched.in_cum_reward_imag = bool(term_next_imag), node_origin.in_cum_reward_imag + float(R_next_imag) * discount_prod
        if flag_record and env_root is not None:
            node_unbranched.env_clone = copy.deepcopy(node_origin.env_clone)
            obs_next, R_next_true, node_unbranched.term_true, _ = node_unbranched.env_clone.step(action_chosen)
            node_unbranched.in_cum_reward_true, node_unbranched.obs = node_origin.in_cum_reward_true + float(not node_origin.term_true) * R_next_true * discount_prod, np.expand_dims(obs_next, 0)
            mask_change = mask_change_minigrid(node_origin.obs, node_unbranched.obs)
            if M.conscious:
                acc_picks_overall, acc_picks_average = _get_acc_compress(weights_attention, mask_change)
                acc_picks_overall, acc_picks_average = acc_picks_overall.numpy(), acc_picks_average.numpy()
                if acc_picks_overall > 0:
                    list_acc_picks_overall.append(acc_picks_overall)
                    list_acc_picks_average.append(acc_picks_average)
                if func_record_image is not None and R_next_true > 0 and node_unbranched.term_true:
                    env_clone = copy.deepcopy(node_origin.env_clone)
                    mask_highlight, flag_highlight = _get_highlight(node_origin.U_imag, tf.constant([action_chosen]), tf.constant(mask_change), M)
                    func_record_image('%s/vis_attention_%d' % (prefix, action_chosen), env_clone.attention_render(highlight_mask=mask_highlight.numpy()), t)
            node_unbranched.mask = tf.logical_or(node_origin.mask, mask_change)
        if node_unbranched.depth > height: height = node_unbranched.depth
        rollouts += 1
    # tree search finished, find the node with best value
    if flag_record:
        if len(POOL_nodes_leaf) and (POOL_values_leaf[-1] > value_current_best_nonleaf or len(POOL_priorities) == 0):
            node_best, value_selected = POOL_nodes_leaf[-1], POOL_values_leaf[-1]
        else:
            node_best, value_selected = node_current_best_nonleaf, value_current_best_nonleaf
    else:
        if len(POOL_nodes_leaf) and (POOL_values_leaf[-1] > value_current_best_nonleaf or len(POOL_priorities) == 0):
            node_best = POOL_nodes_leaf[-1]
        else:
            node_best = node_current_best_nonleaf

    # backtracking action to root, find the best action
    action_best = action_current_best_nonleaf if node_best._is_root else node_best.action_backtrack
    if flag_record:
        func_record_scalar('%s/depth_max' % (prefix), height, t)
        func_record_scalar('%s/agreement' % (prefix), action_intuitive == action_best, t)
        func_record_scalar('%s/depth_stop_selected' % (prefix), node_best.depth, t)
        if env_root is not None:
            if not node_best._is_root:
                node_best.U_true = E(node_best.obs)
                diff_in_cum_reward = np.abs(node_best.in_cum_reward_true - node_best.in_cum_reward_imag)
                func_record_scalar('%s/diff_reward_cum_selected' % (prefix), diff_in_cum_reward, t)
                diff_U = tf.abs(node_best.U_true - node_best.U_imag)
                diff_U_significant = diff_U if bow else diff_U[:, :, 0: 3]
                diff_U_elementwise = tf.reduce_mean(diff_U_significant)
                diff_U_elementwise_changed_objects = tf.reduce_mean(tf.boolean_mask(diff_U_significant, node_best.mask))
                diff_U_elementwise_unchanged_objects = tf.reduce_mean(tf.boolean_mask(diff_U_significant, tf.logical_not(node_best.mask)))
                func_record_scalar('%s/diff_feature_elementwise_selected' % (prefix), diff_U_elementwise, t)
                if not tf.math.is_nan(diff_U_elementwise_changed_objects): func_record_scalar('%s/diff_U_elementwise_changed_objects' % (prefix), diff_U_elementwise_changed_objects, t)
                if not tf.math.is_nan(diff_U_elementwise_unchanged_objects): func_record_scalar('%s/diff_U_elementwise_unchanged_objects' % (prefix), diff_U_elementwise_unchanged_objects, t)
            if node_best.term_imag:
                diff_value = tf.abs(node_best.in_cum_reward_true - value_selected)
            else:
                Q_max_node_best = _get_q_max_from_U(node_best.U_true, Q)
                diff_value = tf.abs(node_best.in_cum_reward_true + Q_max_node_best - value_selected)
            func_record_scalar('%s/diff_value_selected' % (prefix), diff_value, t)
            if M.conscious and len(list_acc_picks_overall):
                func_record_scalar('%s/acc_picks_overall' % (prefix), np.mean(list_acc_picks_overall), t)
                func_record_scalar('%s/acc_picks_average' % (prefix), np.mean(list_acc_picks_average), t)
                if func_record_image is not None:
                    env_clone = copy.deepcopy(node_root.env_clone)
                    obs_next = np.expand_dims(env_clone.step(action_best)[0], 0)
                    mask_change = mask_change_minigrid(node_root.obs, obs_next)
                    mask_highlight, flag_highlight = _get_highlight(node_root.U_imag, tf.constant([action_best]), tf.constant(mask_change), M)
                    if flag_highlight.numpy():
                        func_record_image('%s/vis_attention_%d' % (prefix, action_best), env_root.attention_render(highlight_mask=mask_highlight.numpy()), t)
    return action_best
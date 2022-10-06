from __future__ import print_function
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from util import *


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def attention_aggregation(queries,
                          keys,
                          values,
                          num_unit=None,
                          dropout_rate=0,
                          is_training=True,
                          scope="attention_aggregation",
                          reuse=None
                          ):
    '''
    applies attention aggregation.

    Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        values: A 3d tensor with shape of [N, T_v, C_v].
        num_unit: A scalar, attention size.
        dropout_rate: A float point number
        is_training: Boolean. Controller of mechanism for dropout.
        causality: Boolean. If true, units that reference the future are masked.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        with_qk: Boolean, whether to return Q, K table.

    Returns: A 3d tensor with shape of (N, T_q, C)

    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_unit is None:
            num_unit = queries.get_shape().as_list[-1]

        Q = tf.layers.dense(queries, num_unit, activation=None)  # (N, T_q, C)
        K_1 = tf.layers.dense(keys, num_unit, activation=None)  # (N, T_k, C)
        K_2 = tf.layers.dense(keys, num_unit, activation=None)  # (N, T_k, C)
        V_1 = tf.layers.dense(values, num_unit, activation=None)  # (N, T_v/T_k, C), key is equal to value.
        V_2 = tf.layers.dense(values, num_unit, activation=None)
        score_1 = tf.reshape(tf.reduce_sum(Q * K_1, axis=-1), [tf.shape(Q)[0], tf.shape(Q)[1], 1])
        score_2 = tf.reshape(tf.reduce_sum(Q * K_2, axis=-1), [tf.shape(Q)[0], tf.shape(Q)[1], 1])
        score = tf.concat([score_1, score_2], axis=-1) / (Q.get_shape().as_list()[-1] ** 0.5)
        score = tf.nn.softmax(score)
        score_1, score_2 = tf.split(score, num_or_size_splits=2, axis=-1)
        # outputs = score_1 * V_1 + score_2 * V_2

    return K_1 + K_2


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       # initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.keras.regularizers.l2(l=l2_reg)
                                       )
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs


def multihead_attention(queries,
                        keys,
                        dis_matrix_K,
                        dis_matrix_V,
                        absolute_pos_K,
                        absolute_pos_V,
                        time_matrix_K=None,
                        time_matrix_V=None,
                        longtime_feature=None,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        with_qk=False,
                        ):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        Q = tf.concat(tf.split(queries, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K = tf.concat(tf.split(keys, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V = tf.concat(tf.split(keys, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        Q_ = tf.layers.dense(Q, num_units // num_heads, activation=None)  # (N, T_q, C)
        K_ = tf.layers.dense(K, num_units // num_heads, activation=None)  # (N, T_k, C)
        V_ = tf.layers.dense(V, num_units // num_heads, activation=None)  # (N, T_k, C)
        #### test #####
        weight = tf.get_variable(name='test_weight', shape=[1, 1, num_units])
        weight = tf.tile(weight, [tf.shape(Q)[0], tf.shape(Q)[1], 1])

        # Split and concat
        time_matrix_K_ = tf.concat(tf.split(time_matrix_K, num_heads, axis=3), axis=0)
        time_matrix_V_ = tf.concat(tf.split(time_matrix_V, num_heads, axis=3), axis=0)

        dis_matrix_K_ = tf.concat(tf.split(dis_matrix_K, num_heads, axis=3), axis=0)
        dis_matrix_V_ = tf.concat(tf.split(dis_matrix_V, num_heads, axis=3), axis=0)

        absolute_pos_K_ = tf.concat(tf.split(absolute_pos_K, num_heads, axis=2), axis=0)
        absolute_pos_V_ = tf.concat(tf.split(absolute_pos_V, num_heads, axis=2), axis=0)

        if longtime_feature is not None:
            longtime_K_ = tf.concat(tf.split(longtime_feature, num_heads, axis=2), axis=0)
            longtime_V_ = tf.concat(tf.split(longtime_feature, num_heads, axis=2), axis=0)
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        outputs_pos = tf.matmul(Q_, tf.transpose(absolute_pos_K_, [0, 2, 1]))
        if longtime_feature is not None:
            outputs_longtime = tf.matmul(Q_, tf.transpose(longtime_K_, [0, 2, 1]))
        outputs_time = tf.squeeze(tf.matmul(time_matrix_K_, tf.expand_dims(Q_, axis=3)))
        outputs_dis = tf.squeeze(tf.matmul(dis_matrix_K_, tf.expand_dims(Q_, axis=3)))
        if longtime_feature is not None:
            outputs += outputs_longtime
        outputs += outputs_time
        outputs = outputs + outputs_pos + outputs_dis

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking   padding mask
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)
        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum

        outputs_value = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
        outputs_pos_value = tf.matmul(outputs, absolute_pos_V_)
        if longtime_feature is not None:
            outputs_longtime_value = tf.matmul(outputs, longtime_V_)
        output_time_value = tf.reshape(tf.matmul(tf.expand_dims(outputs, axis=2), time_matrix_V_),
                                       [tf.shape(outputs_pos)[0], tf.shape(outputs_pos)[1], num_units // num_heads])
        output_dis_value = tf.reshape(tf.matmul(tf.expand_dims(outputs, axis=2), dis_matrix_V_),
                                      [tf.shape(outputs_pos)[0], tf.shape(outputs_pos)[1], num_units // num_heads])

        outputs = outputs_value + outputs_pos_value + output_dis_value
        if longtime_feature is not None:
            outputs += outputs_longtime_value
        outputs += output_time_value
        outputs = feedforward(normalize(outputs), num_units=[num_units, num_units], dropout_rate=dropout_rate,
                              num_head=num_heads, is_training=is_training)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                num_head=1,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer

        # inputs = tf.concat(tf.split(inputs, num_head, axis=2), axis=0)
        params = {"inputs": inputs, "filters": num_units[0] // num_head, "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1] // num_head, "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs
        # outputs =  tf.concat(tf.split(outputs, num_head, axis=0), axis=2 )

        # Normalize
        # outputs = normalize(outputs)

    return outputs


def ActivateHelp(data, method):
    if method == 'relu':
        ret = tf.nn.relu(data)
    elif method == 'sigmoid':
        ret = tf.nn.sigmoid(data)
    elif method == 'tanh':
        ret = tf.nn.tanh(data)
    elif method == 'softmax':
        ret = tf.nn.softmax(data, axis=-1)
    elif method == 'leakyRelu':
        ret = tf.maximum(leaky * data, data)
    elif method == 'twoWayLeakyRelu6':
        temMask = tf.to_float(tf.greater(data, 6.0))
        ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * tf.maximum(leaky * data, data)
    elif method == '-1relu':
        ret = tf.maximum(-1.0, data)
    elif method == 'relu6':
        ret = tf.maximum(0.0, tf.minimum(6.0, data))
    elif method == 'relu3':
        ret = tf.maximum(0.0, tf.minimum(3.0, data))
    else:
        raise Exception('Error Activation Function')
    return ret


paramId = 0


def getParamId():
    global paramId
    paramId += 1
    return paramId


def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
    global params
    global regParams
    assert name not in params, 'name %s already exists' % name
    if initializer == 'xavier':
        ret = tf.get_variable(name=name, dtype=dtype, shape=shape,
                              # initializer=xavier_initializer(dtype=tf.float32),
                              initializer=tf.keras.initializers.glorot_normal(dtype=tf.float32),
                              trainable=trainable)
    elif initializer == 'trunc_normal':
        ret = tf.get_variable(name=name,
                              initializer=tf.random.truncated_normal(shape=[int(shape[0]), shape[1]], mean=0.0,
                                                                     stddev=0.03, dtype=dtype))
    elif initializer == 'zeros':
        ret = tf.get_variable(name=name, dtype=dtype,
                              initializer=tf.zeros(shape=shape, dtype=tf.float32),
                              trainable=trainable)
    elif initializer == 'ones':
        ret = tf.get_variable(name=name, dtype=dtype, initializer=tf.ones(shape=shape, dtype=tf.float32),
                              trainable=trainable)
    elif not isinstance(initializer, str):
        ret = tf.get_variable(name=name, dtype=dtype,
                              initializer=initializer, trainable=trainable)
    else:
        print('ERROR: Unrecognized initializer')
        exit()
    params[name] = ret
    if reg:
        regParams[name] = ret
    return ret


params = {}
regParams = {}


def getOrDefineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True, reuse=False):
    global params
    global regParams
    if name in params:
        assert reuse, 'Reusing Param %s Not Specified' % name
        if reg and name not in regParams:
            regParams[name] = params[name]
        return params[name]
    return defineParam(name, shape, dtype, reg, initializer, trainable)


def Bias(data, name=None, reg=False, reuse=False, initializer='zeros'):
    inDim = data.get_shape()[-1]
    temName = name if name != None else 'defaultParamName%d' % getParamId()
    temBiasName = temName + 'Bias'
    bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer=initializer, reuse=reuse)
    if reg:
        regParams[temBiasName] = bias
    return data + bias


def BN(inp):
    global ita
    dim = inp.get_shape()[1]
    scale = tf.Variable(tf.ones([dim]))
    shift = tf.Variable(tf.zeros([dim]))
    fcMean, fcVar = tf.nn.moments(inp, axes=[0])
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    emaApplyOp = ema.apply([fcMean, fcVar])
    with tf.control_dependencies([emaApplyOp]):
        mean = tf.identity(fcMean)
        var = tf.identity(fcVar)
    ret = tf.nn.batch_normalization(inp, mean, var, shift,
                                    scale, 1e-8)
    return ret


def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None,
       initializer='xavier', reuse=False, biasReg=False, biasInitializer='zeros'):
    global params
    global regParams
    global leaky
    inDim = inp.get_shape()[1]
    temName = name if name != None else 'defaultParamName%d' % getParamId()
    W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse)
    if dropout != None:
        ret = tf.nn.dropout(inp, rate=dropout) @ W
    else:
        ret = inp @ W
    if useBias:
        ret = Bias(ret, name=name, reuse=reuse, reg=biasReg, initializer=biasInitializer)
    if useBN:
        ret = BN(ret)
    if activation != None:
        ret = Activate(ret, activation)
    return ret


leaky = 0.1


def Activate(data, method):
    global leaky
    ret = data
    ret = ActivateHelp(ret, method)
    return ret


def hyperPropagate(lats, adj, hyperNum, actFunc):
    lat1 = Activate(tf.transpose(adj) @ lats, actFunc)
    lat2 = tf.transpose(FC(tf.transpose(lat1), hyperNum, activation=actFunc)) + lat1
    lat3 = tf.transpose(FC(tf.transpose(lat2), hyperNum, activation=actFunc)) + lat2
    lat4 = tf.transpose(FC(tf.transpose(lat3), hyperNum, activation=actFunc)) + lat3
    ret = Activate(adj @ lat4, actFunc)
    # ret = adj @ lat4
    return ret


def hyperGraph(embedding_final_dis, embedding_final_tra, w_value, args, keepRate, scope='hyperGraph', g_name='dis',
               reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        embedding_hyper_base_dis = []
        hyper_w = defineParam('hyper_w' + g_name, [args.hidden_units, args.hyperNum], reg=True)
        W_h = []
        w_h0 = w_value
        W_h.append(w_h0)
        weight_sum_emb = w_h0 * embedding_final_dis[0] + (1 - w_h0) * embedding_final_tra[0]
        hyper_dep = tf.matmul(weight_sum_emb, hyper_w)
        for i in range(args.gcn_layers):
            W_h.append(w_value)
            weight_sum_emb_ = W_h[i + 1] * embedding_final_dis[i] + (1 - W_h[i + 1]) * embedding_final_tra[
                i]
            hyper_base_dis = hyperPropagate(weight_sum_emb_, tf.nn.dropout(hyper_dep, keepRate), args.hyperNum,
                                            args.actFunc)
            embedding_hyper_base_dis.append(hyper_base_dis)
        embedding_final_hyper_base_dis = tf.stack(embedding_hyper_base_dis, 1)
        embedding_final_hyper_base_dis = tf.reduce_sum(embedding_final_hyper_base_dis, 1)
    return embedding_final_hyper_base_dis

from modules import *
from util import *


class Model():
    def __init__(self, usernum, itemnum, max_origin_seq_len, timenum, args, num_batch, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.input_origin_seq = tf.placeholder(tf.int32, shape=(None, max_origin_seq_len))
        self.time_matrix = tf.placeholder(tf.int32, shape=(None, args.maxlen, args.maxlen))
        self.dis_matrix = tf.placeholder(tf.int32, shape=(None, args.maxlen, args.maxlen))
        self.dis_matrix_lat = tf.placeholder(tf.int32, shape=(None, args.maxlen, args.maxlen))
        self.dis_matrix_lon = tf.placeholder(tf.int32, shape=(None, args.maxlen, args.maxlen))
        self.global_ = tf.placeholder(dtype=tf.int32)

        self.no_concat_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen),
                                            name='placeholder_no_concat_seq')
        mask_avg = tf.expand_dims(tf.to_float(tf.not_equal(self.no_concat_seq, 0)), -1)


        batch_size = tf.shape(self.input_seq)[0]
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.keepRate = tf.placeholder(dtype=tf.float32, shape=(None))
        pos = self.pos
        neg = self.neg
        self.num_batch = num_batch
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        self.mask_1 = mask
        self.time_matrix = tf.reshape(self.time_matrix,
                                      [tf.shape(self.input_seq)[0], args.maxlen, args.maxlen])
        self.dis_matrix = tf.reshape(self.dis_matrix, [tf.shape(self.input_seq)[0], args.maxlen, args.maxlen])
        self.dis_matrix_lat = tf.reshape(self.dis_matrix_lat, [tf.shape(self.input_seq)[0], args.maxlen, args.maxlen])
        self.dis_matrix_lon = tf.reshape(self.dis_matrix_lon, [tf.shape(self.input_seq)[0], args.maxlen, args.maxlen])
        self.adj_matrix = tf.sparse_placeholder(tf.float32, name='adj_matrix')
        self.tra_matrix = tf.sparse_placeholder(tf.float32, name='tran_matrix')

        adj_matrix_dropout = self.node_dropout(self.adj_matrix, tf.shape(self.adj_matrix.values)[0],
                                               0.8)

        tra_matrix_dropout = self.node_dropout(self.tra_matrix, tf.shape(self.tra_matrix.values)[0],
                                               0.8)
        # adj_matrix_dropout = tf.sparse_tensor_to_dense(adj_matrix_dropout)
        # tra_matrix_dropout = tf.sparse_tensor_to_dense(tra_matrix_dropout)
        #### test ####
        with tf.variable_scope("item_emb", reuse=reuse):
            _, item_emb = embedding(self.input_seq,
                                    vocab_size=itemnum,
                                    num_units=args.hidden_units,
                                    zero_pad=False,
                                    scale=True,
                                    l2_reg=args.l2_emb,
                                    scope="input_embeddings",
                                    with_t=True,
                                    reuse=reuse)
        self.item_emb = item_emb
        with tf.variable_scope("Dis_GCN", reuse=reuse):
            embedding_final = [item_emb]
            layer = item_emb

            matrix_len = tf.shape(layer)[0]
            new_hidden_len = args.hidden_units // args.num_heads
            W = []
            for k in range(args.gcn_layers):  # num of gcn layer
                W.append(tf.get_variable(name='w' + str(k), shape=[args.num_heads, new_hidden_len, new_hidden_len],
                                         regularizer=tf.keras.regularizers.l2(l=args.l2_emb)))
                layer = tf.sparse_tensor_dense_matmul(adj_matrix_dropout, layer)
                # layer = tf.matmul(adj_matrix_dropout, layer)
                layer = tf.reshape(tf.concat(tf.split(layer, args.num_heads, axis=-1), axis=0),
                                   shape=[args.num_heads, matrix_len, -1])
                layer = tf.nn.tanh(tf.matmul(layer, W[k]))
                layer = tf.squeeze(tf.concat(tf.split(layer, args.num_heads, axis=0), axis=-1))
                layer = tf.nn.dropout(layer, keep_prob=1)
                embedding_final += [layer]
            embedding_final_dis = embedding_final
            embedding_final = tf.stack(embedding_final, 1)
            embedding_final = tf.reduce_sum(embedding_final, 1)
        self.debug1 = embedding_final
        with tf.variable_scope("Tra_GCN", reuse=reuse):
            embedding_final_1 = [item_emb]
            layer_1 = item_emb

            matrix_len_1 = tf.shape(layer_1)[0]
            new_hidden_len = args.hidden_units // args.num_heads
            W = []
            for k in range(args.gcn_layers):
                W.append(tf.get_variable(name='w' + str(k), shape=[args.num_heads, new_hidden_len, new_hidden_len],
                                         regularizer=tf.keras.regularizers.l2(l=args.l2_emb)))
                layer_1 = tf.sparse_tensor_dense_matmul(tra_matrix_dropout, layer_1)
                # layer_1 = tf.matmul(tra_matrix_dropout, layer_1)
                layer_1 = tf.reshape(tf.concat(tf.split(layer_1, args.num_heads, axis=-1), axis=0),
                                     shape=[args.num_heads, matrix_len_1, -1])
                layer_1 = tf.nn.tanh(tf.matmul(layer_1, W[k]))
                layer_1 = tf.squeeze(tf.concat(tf.split(layer_1, args.num_heads, axis=0), axis=-1))
                layer_1 = tf.nn.dropout(layer_1, keep_prob=1)
                embedding_final_1 += [layer_1]
            embedding_final_tra = embedding_final_1
            embedding_final_1 = tf.stack(embedding_final_1, 1)
            embedding_final_1 = tf.reduce_sum(embedding_final_1, 1)
        self.debug2 = embedding_final_1
        w_t = tf.get_variable('w_t1', shape=(1), regularizer=tf.keras.regularizers.l2(l=args.l2_emb))
        embedding_final = w_t * embedding_final + (1 - w_t) * embedding_final_1

        embedding_final_hyper_base_dis = hyperGraph(embedding_final_dis, embedding_final_tra, 1, args,
                                                    self.keepRate, scope='hyperGraph_dis', g_name='dis')
        embedding_final_hyper_base_tra = hyperGraph(embedding_final_dis, embedding_final_tra, 0, args,
                                                    self.keepRate, scope='hyperGraph_tra', g_name='tra')
        self.embedding_final_hyper_base_dis = embedding_final_hyper_base_dis
        w_th = tf.get_variable('w_th', shape=(1), regularizer=tf.keras.regularizers.l2(l=args.l2_emb))
        embedding_final_hyper = (1.0 - w_th) * embedding_final_hyper_base_dis + w_th * embedding_final_hyper_base_tra
        if args.W_hyper > 5:
            w_t2 = tf.get_variable('w_t2', shape=(1), regularizer=tf.keras.regularizers.l2(l=args.l2_emb))
            embedding_final = (1.0 - w_t2) * embedding_final + w_t2 * embedding_final_hyper
        else:
            embedding_final = (1.0 - args.W_hyper) * embedding_final + args.W_hyper * embedding_final_hyper


        self.embedding_final_1 = embedding_final
        embedding_final = tf.concat((tf.zeros(shape=[1, args.hidden_units]), embedding_final[:, :]), 0)
        self.embedding_final_2 = embedding_final
        self.emb = embedding_final

        with tf.variable_scope("multi_head_self_attention", reuse=reuse):
            self.seq = tf.nn.embedding_lookup(embedding_final, self.input_seq)
            if args.random_poi_emb:
                self.seq = embedding(
                    self.input_seq,
                    vocab_size=itemnum + 1,
                    num_units=args.hidden_units,
                    zero_pad=False,
                    scale=False,
                    l2_reg=args.l2_emb,
                    scope="seq_emb",
                    reuse=reuse,
                    with_t=False
                )
            masked_seq_feature = self.seq * mask
            avg_feature = tf.reduce_sum(masked_seq_feature, axis=1)
            poi_num = tf.tile(tf.reduce_sum(mask, axis=1), [1, args.hidden_units])
            avg_feature = avg_feature / poi_num
            self.origin_seq_emb = tf.nn.embedding_lookup(embedding_final, self.input_origin_seq)

            latest_poi_emb = tf.expand_dims(avg_feature, 1)
            origin_seq_dot = tf.matmul(latest_poi_emb,
                                       tf.transpose(self.origin_seq_emb, (0, 2, 1)))
            origin_seq_dot = tf.reshape(origin_seq_dot, [batch_size, -1])
            latest_poi_l2 = tf.norm(latest_poi_emb, ord=2, axis=2)
            origin_seq_l2 = tf.norm(self.origin_seq_emb, ord=2, axis=2)
            self.origin_seq_cosine_similarity_avg = (origin_seq_dot / (latest_poi_l2 * origin_seq_l2))

            item_emb_table = embedding_final
            # position mebedding
            absolute_pos_K, absolute_pos_K_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="abs_pos_K",
                reuse=reuse,
                with_t=True
            )
            absolute_pos_V, absolute_pos_V_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="abs_pos_V",
                reuse=reuse,
                with_t=True
            )

            self.absolute_pos_K_1 = absolute_pos_K
            self.absolute_pos_V_1 = absolute_pos_V
            # Time Encoding
            time_matrix_emb_K, time_matrix_emb_K_table = embedding(
                self.time_matrix,
                vocab_size=args.time_span + 1,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_time_K",
                reuse=tf.AUTO_REUSE,
                with_t=True
            )
            time_matrix_emb_V, time_matrix_emb_V_table = embedding(
                self.time_matrix,
                vocab_size=args.time_span + 1,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_time_V",
                reuse=tf.AUTO_REUSE,
                with_t=True
            )

            dis_matrix_lat_emb_K = embedding(
                self.dis_matrix_lat,
                vocab_size=args.time_span * 2 + 1,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_dis_lat_K",
                reuse=reuse,
                with_t=False
            )
            dis_matrix_lon_emb_K = embedding(
                self.dis_matrix_lon,
                vocab_size=args.time_span * 2 + 1,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_dis_lon_K",
                reuse=reuse,
                with_t=False
            )

            dis_matrix_lat_emb_V = embedding(
                self.dis_matrix_lat,
                vocab_size=args.time_span * 2 + 1,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_dis_lat_V",
                reuse=reuse,
                with_t=False
            )
            dis_matrix_lon_emb_V = embedding(
                self.dis_matrix_lon,
                vocab_size=args.time_span * 2 + 1,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_dis_lon_V",
                reuse=reuse,
                with_t=False
            )
            dis_matrix_emb_K = dis_matrix_lat_emb_K + dis_matrix_lon_emb_K
            dis_matrix_emb_V = dis_matrix_lat_emb_V + dis_matrix_lon_emb_V


            self.dis_matrix_emb_K_1 = dis_matrix_emb_K
            self.dis_matrix_emb_V_1 = dis_matrix_emb_V

            # Dropout
            self.seq_1 = self.seq
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq_2 = self.seq
            self.seq *= mask
            self.seq_3 = self.seq
            time_matrix_emb_K = tf.layers.dropout(time_matrix_emb_K,
                                                  rate=args.dropout_rate,
                                                  training=tf.convert_to_tensor(self.is_training))
            time_matrix_emb_V = tf.layers.dropout(time_matrix_emb_V,
                                                  rate=args.dropout_rate,
                                                  training=tf.convert_to_tensor(self.is_training))
            dis_matrix_emb_K = tf.layers.dropout(dis_matrix_emb_K,
                                                 rate=args.dropout_rate,
                                                 training=tf.convert_to_tensor(self.is_training))
            dis_matrix_emb_V = tf.layers.dropout(dis_matrix_emb_V,
                                                 rate=args.dropout_rate,
                                                 training=tf.convert_to_tensor(self.is_training))

            absolute_pos_K = tf.layers.dropout(absolute_pos_K,
                                               rate=args.dropout_rate,
                                               training=tf.convert_to_tensor(self.is_training))
            absolute_pos_V = tf.layers.dropout(absolute_pos_V,
                                               rate=args.dropout_rate,
                                               training=tf.convert_to_tensor(self.is_training))
            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   time_matrix_K=time_matrix_emb_K,
                                                   time_matrix_V=time_matrix_emb_V,
                                                   dis_matrix_K=dis_matrix_emb_K,
                                                   dis_matrix_V=dis_matrix_emb_V,
                                                   absolute_pos_K=absolute_pos_K,
                                                   absolute_pos_V=absolute_pos_V,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention",
                                                   reuse=tf.AUTO_REUSE
                                                   )
            self.seq_emb333 = self.seq
            self.seq *= mask_avg
            self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        seq_emb_1 = tf.reshape(seq_emb, [tf.shape(self.input_seq)[0], args.maxlen, args.hidden_units])
        self.test_logits = tf.matmul(seq_emb_1, tf.transpose(item_emb_table))
        self.test_logits = tf.reshape(self.test_logits,
                                      [tf.shape(self.input_seq)[0], args.maxlen, -1])
        self.test_logits = tf.squeeze(self.test_logits[:, -1, :])

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        ###### test ######
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        all_score = tf.matmul(seq_emb, item_emb_table, transpose_b=True)

        padding = -tf.ones_like(pos)

        self.loss_classify = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pos, logits=all_score) * istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        self.loss_classify += sum(reg_losses)

        tf.summary.scalar('loss', self.loss_classify)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.train.exponential_decay(args.lr, self.global_,
                                                            15, 0.9, staircase=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss_classify, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, args, u, seq, time_matrix, time_matrix_origin, dis_matrix, dis_matrix_lat, dis_matrix_lon,
                item_idx, adj_matrix,
                tra_matrix, origin_seq=None, origin_time_seq=None, origin_seq_location=None, max_origin_seq_len=None,
                all_poi_dis_matrix=None, all_poi_dis_matrix_lat=None, all_poi_dis_matrix_lon=None, diagonal_zero=None,
                global_=None):
        cosine_similarity_avg = sess.run([self.origin_seq_cosine_similarity_avg],
                                         {self.u: u, self.input_seq: seq,
                                          self.input_origin_seq: origin_seq,
                                          self.time_matrix: time_matrix,
                                          self.global_: global_,
                                          self.dis_matrix: dis_matrix,
                                          self.dis_matrix_lat: dis_matrix_lat,
                                          self.dis_matrix_lon: dis_matrix_lon,
                                          self.adj_matrix: adj_matrix,
                                          self.keepRate: 0.5, self.tra_matrix: tra_matrix,
                                          self.is_training: False})

        cosine_similarity_avg = np.nan_to_num(cosine_similarity_avg)[0]
        batch_size = len(u)
        diagonal_zero_for_avg = np.ones([1, max_origin_seq_len])
        diagonal_zero_for_avg[:, -args.maxlen_origin:] = 0
        cosine_similarity_avg = cosine_similarity_avg * diagonal_zero_for_avg
        cosine_similarity_avg = cosine_similarity_avg.astype(np.float64)
        tt = cosine_similarity_avg.argsort()
        sorted_poi_idx = tt[:, -args.max_longtime_avg_len:]
        sorted_poi_idx = np.sort(sorted_poi_idx, axis=1)

        longtime_seq = np.zeros([batch_size, args.max_longtime_avg_len], dtype=np.int32)
        num_matrix = np.ones([batch_size]) * int(args.max_longtime_avg_len - 1)
        similar_poi_idx_dict = {}
        for b in range(batch_size):
            for location in range(args.max_longtime_avg_len):
                index = args.max_longtime_avg_len - location - 1
                batch_idx = b
                if batch_idx not in similar_poi_idx_dict:
                    similar_poi_idx_dict[batch_idx] = []
                similar_poi_idx = sorted_poi_idx[b][index]
                if cosine_similarity_avg[batch_idx][similar_poi_idx] <= args.similar_rate_avg:
                    continue
                if similar_poi_idx >= max_origin_seq_len - args.maxlen_origin - 1 or num_matrix[batch_idx] < 0:
                    continue

                similar_poi_idx_dict[batch_idx].append(similar_poi_idx)
                idx_ = int(num_matrix[batch_idx])
                longtime_seq[batch_idx][idx_] = origin_seq[batch_idx][similar_poi_idx]
                num_matrix[batch_idx] -= 1
        old_seq, longtime_dis_matrix, longtime_dis_matrix_lat, longtime_dis_matrix_lon, longtime_time_matrix = \
            concat_similar_poi(longtime_seq, seq, origin_seq, dis_matrix, dis_matrix_lat,
                               dis_matrix_lon,
                               time_matrix, time_matrix_origin, all_poi_dis_matrix,
                               all_poi_dis_matrix_lat,
                               all_poi_dis_matrix_lon, args, similar_poi_idx_dict)
        span_matrix = np.ones_like(longtime_dis_matrix_lat) * args.time_span
        longtime_dis_matrix_lat = longtime_dis_matrix_lat + span_matrix
        longtime_dis_matrix_lon = longtime_dis_matrix_lon + span_matrix
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: old_seq, self.time_matrix: longtime_time_matrix,
                         self.dis_matrix: longtime_dis_matrix,
                         self.dis_matrix_lat: longtime_dis_matrix_lat, self.dis_matrix_lon: longtime_dis_matrix_lon,
                         self.global_: global_, self.no_concat_seq: seq,
                         self.adj_matrix: adj_matrix, self.tra_matrix: tra_matrix, self.is_training: False,
                         self.keepRate: 0.5})

    def node_dropout(self, adj_matrix, num_value, keep_prob):
        noise_shape = [num_value]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(adj_matrix, dropout_mask)

        return pre_out

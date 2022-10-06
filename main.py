import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model import Model
from util import *
import pickle


np.set_printoptions(threshold=np.inf)
print('gpu available: ', tf.test.is_gpu_available())
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

random.seed(2022)
np.random.seed(2022)
tf.set_random_seed(2022)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Gowalla', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--test_batch_size', default=64, type=int)
parser.add_argument('--train_filter', default=4, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.4, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--hyperNum', default=16, type=int, help='number of hyper edges')
parser.add_argument('--W_hyper', default=10, type=float, help='if W_hyper > 5, then use weighted hypergraph')
parser.add_argument('--similar_rate_avg', default=0.9, type=float)
parser.add_argument('--max_longtime_avg_len', default=3, type=int)
parser.add_argument('--time_span', default=256, type=int)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--maxlen_origin', default=97, type=int)
parser.add_argument('--hidden_units', default=128, type=int)
parser.add_argument('--num_epochs', default=501, type=int)
parser.add_argument('--num_heads', default=4, type=int)
parser.add_argument('--gcn_layers', default=3, type=int)
parser.add_argument('--actFunc', default='tanh', type=str)
parser.add_argument('--save_model', default=True, type=bool)
parser.add_argument('--is_debug', default=False, type=bool)
parser.add_argument('--test', default=False, type=bool)

args = parser.parse_args()
args.maxlen_origin = args.maxlen - args.max_longtime_avg_len

path_name = 'lr' + str(args.lr) + '_' + args.dataset

if not os.path.isdir(path_name):
    os.makedirs(path_name)
with open(os.path.join(path_name, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()
dis_matrix_path = './data/' + args.dataset + '/distance_matrix_train.npz'
tra_matrix_path = './data/' + args.dataset + '/transition_matrix_train.npz'
fpath = './poidata/' + args.dataset + '/processed_' + args.dataset + '.txt'

dataset, poi_rating_dict = data_partition(fpath, args)
dataset[-1] = 300
[user_train, user_valid, user_test, usernum, itemnum, timenum, max_origin_seq_len] = dataset
diagonal_zero = np.ones([max_origin_seq_len, max_origin_seq_len])
for i in range(max_origin_seq_len):
    diagonal_zero[i, i] = 0
diagonal_zero[:, -args.maxlen_origin:] = 0

diagonal_zero_for_avg = np.ones([args.batch_size, max_origin_seq_len])
diagonal_zero_for_avg[:, -args.maxlen_origin:] = 0
diagonal_zero_train = np.tile(np.expand_dims(diagonal_zero, 0), [args.batch_size, 1, 1])
num_batch = int(len(user_train) / args.batch_size)
tf.reset_default_graph()
model = Model(usernum, itemnum, max_origin_seq_len, timenum, args, num_batch)
config = tf.ConfigProto()  # config protocol
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

matrix = sp.load_npz(dis_matrix_path)
matrix1 = sp.load_npz(tra_matrix_path)
matrix = get_adj_matrix(matrix)
matrix1 = get_adj_matrix(matrix1)

dataset_name = args.dataset
try:
    relation_matrix = pickle.load(
        open('./poidata/processed_data/%s/relation_matrix_%s_%d_%d.pickle' % (
            dataset_name, dataset_name, args.maxlen, args.time_span), 'rb'))
    relation_matrix_origin = pickle.load(
        open('./poidata/processed_data/%s/relation_matrix_origin_%s_%d_%d.pickle' % (
            dataset_name, dataset_name, args.maxlen, args.time_span), 'rb'))
except Exception as e:
    print('load exception: ', e)
    relation_matrix, relation_matrix_origin = Relation(user_train, usernum, args.maxlen, args.maxlen_origin,
                                                       args.time_span, max_origin_seq_len, args)
    pickle.dump(relation_matrix,
                open('./poidata/processed_data/%s/relation_matrix_%s_%d_%d.pickle' % (
                    dataset_name, dataset_name, args.maxlen, args.time_span), 'wb'))
    pickle.dump(relation_matrix_origin,
                open('./poidata/processed_data/%s/relation_matrix_origin_%s_%d_%d.pickle' % (
                    dataset_name, dataset_name, args.maxlen, args.time_span), 'wb'))

try:
    dis_relation_matrix = pickle.load(
        open('./poidata/processed_data/%s/relation_dis_matrix_%s_%d_%d.pickle' % (
        dataset_name, dataset_name, args.maxlen, args.time_span), 'rb'))
    dis_relation_matrix_lat = pickle.load(
        open('./poidata/processed_data/%s/relation_dis_matrix_lat_%s_%d_%d.pickle' % (
        dataset_name, dataset_name, args.maxlen, args.time_span), 'rb'))
    dis_relation_matrix_lon = pickle.load(
        open('./poidata/processed_data/%s/relation_dis_matrix_lon_%s_%d_%d.pickle' % (
        dataset_name, dataset_name, args.maxlen, args.time_span), 'rb'))
except:
    dis_relation_matrix, dis_relation_matrix_lat, dis_relation_matrix_lon = Relation_dis(user_train, usernum,
                                                                                         args.maxlen,
                                                                                         args.maxlen_origin,
                                                                                         args.time_span, args)
    pickle.dump(dis_relation_matrix,
                open('./poidata/processed_data/%s/relation_dis_matrix_%s_%d_%d.pickle' % (
                dataset_name, dataset_name, args.maxlen, args.time_span), 'wb'))
    pickle.dump(dis_relation_matrix_lat,
                open('./poidata/processed_data/%s/relation_dis_matrix_lat_%s_%d_%d.pickle' % (
                dataset_name, dataset_name, args.maxlen, args.time_span),
                     'wb'))
    pickle.dump(dis_relation_matrix_lon,
                open('./poidata/processed_data/%s/relation_dis_matrix_lon_%s_%d_%d.pickle' % (
                dataset_name, dataset_name, args.maxlen, args.time_span),
                     'wb'))

try:
    all_poi_dis_matrix = pickle.load(
        open('./poidata/processed_data/%s/poi_dis_matrix_%s_%d_%d.pickle' % (
        dataset_name, dataset_name, itemnum, args.time_span), 'rb'))
    all_poi_dis_matrix_lat = pickle.load(
        open('./poidata/processed_data/%s/poi_dis_matrix_lat_%s_%d_%d.pickle' % (
        dataset_name, dataset_name, itemnum, args.time_span), 'rb'))
    all_poi_dis_matrix_lon = pickle.load(
        open('./poidata/processed_data/%s/poi_dis_matrix_lon_%s_%d_%d.pickle' % (
        dataset_name, dataset_name, itemnum, args.time_span), 'rb'))
except:
    print('Generation all poi distance matrix...')
    all_poi_dis_matrix, all_poi_dis_matrix_lat, all_poi_dis_matrix_lon = Poi_dis_matrix(poi_rating_dict, itemnum, args)
    pickle.dump(all_poi_dis_matrix,
                open('./poidata/processed_data/%s/poi_dis_matrix_%s_%d_%d.pickle' % (
                dataset_name, dataset_name, itemnum, args.time_span), 'wb'), protocol=4)
    pickle.dump(all_poi_dis_matrix_lat,
                open('./poidata/processed_data/%s/poi_dis_matrix_lat_%s_%d_%d.pickle' % (
                dataset_name, dataset_name, itemnum, args.time_span), 'wb'), protocol=4)
    pickle.dump(all_poi_dis_matrix_lon,
                open('./poidata/processed_data/%s/poi_dis_matrix_lon_%s_%d_%d.pickle' % (
                dataset_name, dataset_name, itemnum, args.time_span), 'wb'), protocol=4)

sampler = WarpSampler(user_train, usernum, itemnum, max_origin_seq_len, relation_matrix, relation_matrix_origin,
                      dis_relation_matrix,
                      dis_relation_matrix_lat, dis_relation_matrix_lon, args, batch_size=args.batch_size,
                      maxlen=args.maxlen, maxlen_origin=args.maxlen_origin, n_workers=2)
u, seq, origin_seq, origin_seq_location, time_seq, origin_time_seq, time_matrix, time_matrix_origin, dis_matrix, dis_matrix_lat, dis_matrix_lon, pos, neg = sampler.next_batch()

T = 0.0
t0 = time.time()
print('args = ', args)
print('path_name = ', path_name)
dialog_zero_3D = np.ones([args.batch_size, max_origin_seq_len, max_origin_seq_len])
for i in range(max_origin_seq_len):
    dialog_zero_3D[:, i, i] = 0
dialog_zero_2D = np.reshape(dialog_zero_3D, [-1, max_origin_seq_len])
for epoch in range(1, args.num_epochs + 1):
    for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b', desc='epoch ' + str(epoch)):
        u, seq, origin_seq, origin_seq_location, time_seq, origin_time_seq, time_matrix, time_matrix_origin, dis_matrix, dis_matrix_lat, dis_matrix_lon, pos, neg = sampler.next_batch()
        cosine_similarity_avg = sess.run([model.origin_seq_cosine_similarity_avg],
                                         {model.u: u, model.input_seq: seq,
                                          model.input_origin_seq: origin_seq,
                                          model.time_matrix: time_matrix, model.pos: pos,
                                          model.global_: epoch,
                                          model.neg: neg,
                                          model.dis_matrix: dis_matrix,
                                          model.dis_matrix_lat: dis_matrix_lat,
                                          model.dis_matrix_lon: dis_matrix_lon,
                                          model.adj_matrix: matrix,
                                          model.keepRate: 0.5, model.tra_matrix: matrix1,
                                          model.is_training: False})
        cosine_similarity_avg = np.nan_to_num(cosine_similarity_avg)[0]
        cosine_similarity_avg = cosine_similarity_avg * diagonal_zero_for_avg
        cosine_similarity_avg = cosine_similarity_avg.astype(np.float64)
        tt = cosine_similarity_avg.argsort()
        sorted_poi_idx = tt[:, -args.max_longtime_avg_len:]
        sorted_poi_idx = np.sort(sorted_poi_idx, axis=1)

        longtime_seq = np.zeros([args.batch_size, args.max_longtime_avg_len], dtype=np.int32)
        num_matrix = np.ones([args.batch_size]) * int(args.max_longtime_avg_len - 1)
        similar_poi_idx_dict = {}
        for b in range(args.batch_size):
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

        auc, loss, _ = sess.run([model.auc, model.loss_classify, model.train_op],
                                {model.u: u, model.input_seq: old_seq, model.input_origin_seq: origin_seq,
                                 model.global_: epoch, model.no_concat_seq: seq,
                                 model.time_matrix: longtime_time_matrix, model.pos: pos, model.neg: neg,
                                 model.dis_matrix: longtime_dis_matrix, model.dis_matrix_lat: longtime_dis_matrix_lat,
                                 model.dis_matrix_lon: longtime_dis_matrix_lon, model.adj_matrix: matrix,
                                 model.keepRate: 0.5, model.tra_matrix: matrix1, model.is_training: True})
    if epoch % 2 == 0:
        t1 = time.time() - t0
        T += t1
        print('Evaluating')
        t_valid = evaluate_valid(model, dataset, args, sess, matrix, matrix1, all_poi_dis_matrix,
                                 all_poi_dis_matrix_lat, all_poi_dis_matrix_lon, diagonal_zero)
        print('epoch:%d, time: %f(s), valid (NDCG@2: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, HR@2: %.4f,HR@5: %.4f,'
              ' HR@10: %.4f)' % (epoch, T, t_valid[0][0], t_valid[0][1], t_valid[0][2], t_valid[1][0], t_valid[1][1], t_valid[1][2]))
        saver = tf.train.Saver()
        if args.save_model:
            model_path = "./saved_model/" + str(epoch)
            if not os.path.isdir(model_path):
                os.makedirs(model_path + '/model')
            saver.save(sess, model_path + '/model')
        t0 = time.time()

sampler.close()

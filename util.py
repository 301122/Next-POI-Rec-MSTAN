import sys
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt
from multiprocessing import Process, Queue
import scipy.sparse as sp
import pickle
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2.0 * asin(sqrt(a))
    r = 6371.0  # km
    return c * r  # km， must to be postive


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def computeRePos(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def computedisPos(time_seq, time_span):
    time_span = time_span
    size = len(time_seq)
    dis_matrix = np.zeros([size, size], dtype=np.float64)
    dis_matrix_lat = np.zeros([size, size], dtype=np.float64)
    dis_matrix_lon = np.zeros([size, size], dtype=np.float64)
    for i in range(size):
        for j in range(size):
            # print(time_seq)
            lon1 = float(time_seq[i].split(',')[0])
            lat1 = float(time_seq[i].split(',')[1])
            lon2 = float(time_seq[j].split(',')[0])
            lat2 = float(time_seq[j].split(',')[1])
            span = int(abs(haversine(lon1, lat1, lon2, lat2)))
            span_lat = int(abs(haversine(0.0, lat1, 0.0, lat2)))
            span_lon = int(abs(haversine(lon1, 0.0, lon2, 0.0)))
            if span_lat > time_span:
                dis_matrix_lat[i][j] = -time_span if lat2 < lat1 else time_span
            else:
                dis_matrix_lat[i][j] = -span_lat if lat2 < lat1 else span_lat
            if span_lon > time_span:
                dis_matrix_lon[i][j] = -time_span if lon2 < lon1 else time_span
            else:
                dis_matrix_lon[i][j] = -span_lon if lon2 < lon1 else span_lon
            if span > time_span:
                dis_matrix[i][j] = time_span
            else:
                dis_matrix[i][j] = span
    return dis_matrix, dis_matrix_lat, dis_matrix_lon


def Relation(user_train, usernum, maxlen, maxlen_origin, time_span, max_origin_seq_len, args):
    data_train = dict()
    data_train_origin = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        time_seq_origin = np.zeros([max_origin_seq_len], dtype=np.int32)
        idx = maxlen - 1
        idx_origin = max_origin_seq_len - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == maxlen - maxlen_origin - 1: break
        for i in reversed(user_train[user][:-1]):
            time_seq_origin[idx_origin] = i[1]
            idx_origin -= 1
            if idx_origin == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
        data_train_origin[user] = computeRePos(time_seq_origin, time_span)
    return data_train, data_train_origin


def Relation_dis(user_train, usernum, maxlen, maxlen_origin, time_span, args):
    data_train = dict()
    data_train_lat = dict()
    data_train_lon = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing dis relation matrix'):
        seq = ['0,0'] * maxlen
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[2]
            idx -= 1
            if idx == maxlen - maxlen_origin - 1: break
        data_train[user], data_train_lat[user], data_train_lon[user] = computedisPos(seq, time_span)
    return data_train, data_train_lat, data_train_lon


def Poi_dis_matrix(poi_rating_dict, item_num, args):
    time_span = args.time_span
    dis_matrix = np.zeros([item_num + 1, item_num + 1])
    dis_matrix_lat = np.zeros([item_num + 1, item_num + 1])
    dis_matrix_lon = np.zeros([item_num + 1, item_num + 1])
    for i in tqdm(range(1, item_num + 1), desc='Preparing Poi_dis matrix'):
        for j in range(1, item_num + 1):
            lon1 = float(poi_rating_dict[i].split(',')[0])
            lat1 = float(poi_rating_dict[i].split(',')[1])
            lon2 = float(poi_rating_dict[j].split(',')[0])
            lat2 = float(poi_rating_dict[j].split(',')[1])
            span = int(abs(haversine(lon1, lat1, lon2, lat2)))
            span_lat = int(abs(haversine(0.0, lat1, 0.0, lat2)))
            span_lon = int(abs(haversine(lon1, 0.0, lon2, 0.0)))
            if span_lat > time_span:
                dis_matrix_lat[i][j] = -time_span if lat2 < lat1 else time_span
            else:
                dis_matrix_lat[i][j] = -span_lat if lat2 < lat1 else span_lat
            if span_lon > time_span:
                dis_matrix_lon[i][j] = -time_span if lon2 < lon1 else time_span
            else:
                dis_matrix_lon[i][j] = -span_lon if lon2 < lon1 else span_lon
            if span > time_span:
                dis_matrix[i][j] = time_span
            else:
                dis_matrix[i][j] = span
    return dis_matrix, dis_matrix_lat, dis_matrix_lon


def sample_function(user_train, usernum, itemnum, max_origin_seq_len, batch_size, maxlen, maxlen_origin,
                    relation_matrix, relation_matrix_origin, dis_matrix,
                    dis_matrix_lat, dis_matrix_lon, result_queue, SEED, args):
    def sample(user):
        origin_seq = np.zeros([max_origin_seq_len], dtype=np.int32)
        origin_seq_location = ['0,0'] * max_origin_seq_len
        origin_time_seq = np.zeros([max_origin_seq_len], dtype=np.int32)
        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]

        idx = maxlen - 1
        origin_idx = max_origin_seq_len - 1
        ts = set(map(lambda x: x[0], user_train[user]))
        for i in reversed(user_train[user][:]):
            origin_seq[origin_idx] = i[0]
            origin_time_seq[origin_idx] = i[1]
            origin_seq_location[origin_idx] = i[2]
            origin_idx -= 1
            if origin_idx == -1:
                break
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == maxlen - maxlen_origin - 1: break
        time_matrix = relation_matrix[user]
        time_matrix_origin = relation_matrix_origin[user]
        dis_ma = dis_matrix[user]
        dis_ma_lat = dis_matrix_lat[user]
        dis_ma_lon = dis_matrix_lon[user]
        origin_seq = np.array(origin_seq, dtype=np.int32)
        origin_seq_location = np.array(origin_seq_location, dtype=np.str)
        return (
            user, seq, origin_seq, origin_seq_location, time_seq, origin_time_seq, time_matrix, time_matrix_origin,
            dis_ma,
            dis_ma_lat,
            dis_ma_lon, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= args.train_filter: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, max_origin_seq_len, relation_matrix, relation_matrix_origin, dis_matrix,
                 dis_matrix_lat,
                 dis_matrix_lon, args, batch_size=64, maxlen=103, maxlen_origin=100, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      max_origin_seq_len,
                                                      batch_size,
                                                      maxlen,
                                                      maxlen_origin,
                                                      relation_matrix,
                                                      relation_matrix_origin,
                                                      dis_matrix,
                                                      dis_matrix_lat,
                                                      dis_matrix_lon,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      args
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time - time_min)))
    return time_map, time_min


def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1

    item_map = dict()
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(
            map(lambda x: [item_map[x[0]], time_map[x[1]], x[2]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(
            map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1), x[2]], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))
    poi_rating_dict = {}
    for user, items in User_res.items():
        poi_list = list(map(lambda x: x[0], items))
        rating_list = list(map(lambda x: x[2], items))
        new_dict = dict(zip(poi_list, rating_list))
        poi_rating_dict = {**poi_rating_dict, **new_dict}

    return User_res, len(user_set), len(item_set), max(time_max), poi_rating_dict


def get_sorted_list(d, reverse=False):
    return sorted(d.items(), key=lambda x: x[1], reverse=reverse)


def data_partition(fpath, args):
    User = defaultdict(list)  # key: user id, value: list of [item_id, timestamp, lat, lon]
    user_train = {}
    user_valid = {}
    user_test = {}

    print('Preparing data...')
    f = open(fpath, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:  # user id, item id, latitude, longitude, time stamp
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open(fpath, 'r')
    for line in f:
        u, i, rating, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u] < 5 or item_count[i] < 5:
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp, rating])
    f.close()

    time_map, time_min = timeSlice(time_set)
    User, usernum, itemnum, timenum, poi_rating_dict = cleanAndsort(User, time_map)

    origin_seq_len_dict = dict()

    max_origin_seq_len = 0

    sample_num = 0
    repeat_num = 0
    origin_seq_len_set = set()
    total_check_in_num = 0
    for user in User:
        nfeedback = len(User[user])
        valid_ratio = int(len(User[user]) * 0.8)
        test_ratio = int(len(User[user]) * 0.9)
        check_in = User[user]
        origin_seq_len_set.add(nfeedback)
        if nfeedback not in origin_seq_len_dict:
            origin_seq_len_dict[nfeedback] = 1
        else:
            origin_seq_len_dict[nfeedback] += 1
        if nfeedback > max_origin_seq_len:
            max_origin_seq_len = nfeedback
        for i in range(1, len(check_in)):
            now_poi = int(check_in[i][0])
            for j in range(i):
                his_poi = int(check_in[j][0])
                if now_poi == his_poi:
                    repeat_num += 1
                sample_num += 1
        total_check_in_num += nfeedback
        if nfeedback < 5:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            if not args.use_new_partition:
                user_train[user] = User[user][:valid_ratio]
                user_valid[user] = []
                user_valid[user].append(User[user][valid_ratio:test_ratio])
                user_test[user] = []
                user_test[user].append(User[user][test_ratio:])  #
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2:-1])
                user_test[user] = []
                user_test[user].append(User[user][-1:])  #
    return [user_train, user_valid, user_test, usernum, itemnum, timenum, max_origin_seq_len], poi_rating_dict


def evaluate(model, dataset, args, sess, adj_matrix, tra_matrix, all_poi_dis_matrix, all_poi_dis_matrix_lat,
             all_poi_dis_matrix_lon, diagonal_zero):
    [train, valid, test, usernum, itemnum, timenum, max_origin_seq_len] = copy.deepcopy(dataset)
    NDCG = [0.0, 0.0, 0.0]
    valid_user = 0.0
    HT = [0.0, 0.0, 0.0]
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    path_to_pkl = './data/test_data.pkl'
    try:
        with open(path_to_pkl, 'rb') as f:
            all_u = pickle.load(f)
            all_seqs = pickle.load(f)
            all_time_matrix = pickle.load(f)
            all_time_matrix_origin = pickle.load(f)
            all_distance_matrix = pickle.load(f)
            all_distance_matrix_lat = pickle.load(f)
            all_distance_matrix_lon = pickle.load(f)
            all_item_matrix = pickle.load(f)  #
            all_labels = pickle.load(f)
            all_origin_seqs = pickle.load(f)
            all_origin_time_seqs = pickle.load(f)
            all_origin_dis_seqs = pickle.load(f)
    except:
        print('begin writing test')
        print('path_to_pkl = ', path_to_pkl)
        all_u = []
        all_seqs = []
        all_time_matrix = []
        all_time_matrix_origin = []
        all_distance_matrix = []
        all_distance_matrix_lat = []
        all_distance_matrix_lon = []
        all_item_matrix = []
        all_labels = []
        all_origin_seqs = []
        all_origin_time_seqs = []
        all_origin_dis_seqs = []
        for u in tqdm(users, total=len(users), ncols=120,
                      leave=False, unit='b', desc='Generating test dataset'):
            if len(train[u]) < 1 or len(valid[u]) < 1 or len(test[u]) < 1: continue
            if u % 1000 == 0:
                print('.', end='')
                sys.stdout.flush()

            for j in range(len(test[u][0])):
                seq = np.zeros([args.maxlen], dtype=np.int32)
                time_seq = np.zeros([args.maxlen], dtype=np.int32)
                dis_seq = ['0,0'] * args.maxlen
                idx = args.maxlen - 1

                origin_seq = np.zeros([max_origin_seq_len], dtype=np.int32)
                origin_time_seq = np.zeros([max_origin_seq_len], dtype=np.int32)
                origin_dis_seq = ['0,0'] * max_origin_seq_len
                origin_idx = max_origin_seq_len - 1

                extra_seq = test[u][0][:j]
                all_seq = train[u] + valid[u][0] + extra_seq

                for i in reversed(all_seq):
                    origin_seq[origin_idx] = i[0]
                    origin_time_seq[origin_idx] = i[1]
                    origin_dis_seq[origin_idx] = i[2]
                    origin_idx -= 1
                    if origin_idx == -1:
                        break
                for i in reversed(all_seq):
                    seq[idx] = i[0]
                    time_seq[idx] = i[1]
                    dis_seq[idx] = i[2]
                    idx -= 1
                    if idx == args.maxlen - args.maxlen_origin - 1: break

                rated = set(map(lambda x: x[0], all_seq))
                if test[u][0][j][0] in rated:
                    rated.remove(test[u][0][j][0])
                item_idx = list(rated)
                label = test[u][0][j][0]
                time_matrix = computeRePos(time_seq, args.time_span)
                time_matrix_origin = computeRePos(origin_time_seq, args.time_span)
                dis_matrix, dis_matrix_lat, dis_matrix_lon = computedisPos(dis_seq, args.time_span)
                all_u.append(u)
                all_seqs.append(seq)
                all_time_matrix.append(time_matrix)
                all_time_matrix_origin.append(time_matrix_origin)
                all_distance_matrix.append(dis_matrix)
                all_distance_matrix_lat.append(dis_matrix_lat)
                all_distance_matrix_lon.append(dis_matrix_lon)
                all_item_matrix.append(item_idx)
                all_labels.append(label)
                all_origin_seqs.append(origin_seq)
                all_origin_time_seqs.append(origin_time_seq)
                all_origin_dis_seqs.append(origin_dis_seq)

        with open(path_to_pkl, 'wb') as f:
            pickle.dump(all_u, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_seqs, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_time_matrix, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_time_matrix_origin, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_distance_matrix, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_distance_matrix_lat, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_distance_matrix_lon, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_item_matrix, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_labels, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_origin_seqs, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_origin_time_seqs, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_origin_dis_seqs, f, pickle.HIGHEST_PROTOCOL)

    for i in tqdm(range(len(users) // args.test_batch_size), total=len(users) // args.test_batch_size, ncols=70,
                  leave=False, unit='b', desc='Testing...'):
        if i < len(users) // args.test_batch_size:
            u = all_u[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            seq = all_seqs[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            time_matrix = all_time_matrix[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            time_matrix_origin = all_time_matrix_origin[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            dis_matrix = all_distance_matrix[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            dis_matrix_lat = all_distance_matrix_lat[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            dis_matrix_lon = all_distance_matrix_lon[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            item_idx = all_item_matrix[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            label = all_labels[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            origin_seq = all_origin_seqs[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            origin_time_seq = all_origin_time_seqs[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            origin_dis_seq = all_origin_dis_seqs[i * args.test_batch_size:(i + 1) * args.test_batch_size]
        else:
            u = all_u[i * args.test_batch_size:]
            seq = all_seqs[i * args.test_batch_size:]
            time_matrix = all_time_matrix[i * args.test_batch_size:]
            time_matrix_origin = all_time_matrix_origin[i * args.test_batch_size:]
            dis_matrix = all_distance_matrix[i * args.test_batch_size:]
            dis_matrix_lat = all_distance_matrix_lat[i * args.test_batch_size:]
            dis_matrix_lon = all_distance_matrix_lon[i * args.test_batch_size:]
            item_idx = all_item_matrix[i * args.test_batch_size:]
            label = all_labels[i * args.test_batch_size:]
            origin_seq = all_origin_seqs[i * args.test_batch_size:]
            origin_time_seq = all_origin_time_seqs[i * args.test_batch_size:]
            origin_dis_seq = all_origin_dis_seqs[i * args.test_batch_size:]

        predictions = -model.predict(sess, args, u, seq, time_matrix, time_matrix_origin, dis_matrix, dis_matrix_lat,
                                     dis_matrix_lon,
                                     item_idx, adj_matrix, tra_matrix, origin_seq, origin_time_seq, origin_dis_seq,
                                     max_origin_seq_len, all_poi_dis_matrix, all_poi_dis_matrix_lat,
                                     all_poi_dis_matrix_lon,
                                     diagonal_zero, i)
        mask_matrix = np.ones_like(predictions)
        for i in range(mask_matrix.shape[0]):
            for j in range(len(item_idx[i])):
                if item_idx[i][j] != label[i]:
                    mask_matrix[i, item_idx[i][j]] = 0

        predictions = predictions * mask_matrix
        ranks = predictions.argsort().argsort()
        rank = []
        for i in range(len(ranks)):
            rank.append(ranks[i, label[i]])

        valid_user += len(rank)
        for i in rank:
            if i < 2:
                NDCG[0] += 1 / np.log2(i + 2)
                HT[0] += 1
            if i < 5:
                NDCG[1] += 1 / np.log2(i + 2)
                HT[1] += 1
            if i < 10:
                NDCG[2] += 1 / np.log2(i + 2)
                HT[2] += 1

    return list(map(lambda x: x / valid_user, NDCG)), list(map(lambda x: x / valid_user, HT))


def evaluate_valid(model, dataset, args, sess, adj_matrix, tra_matrix, all_poi_dis_matrix, all_poi_dis_matrix_lat,
                   all_poi_dis_matrix_lon, diagonal_zero):
    [train, valid, test, usernum, itemnum, timenum, max_origin_seq_len] = copy.deepcopy(dataset)
    NDCG = [0.0, 0.0, 0.0]
    valid_user = 0.0
    HT = [0.0, 0.0, 0.0]
    RECALL = [0.0, 0.0, 0.0]
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    path_to_pkl = './data/valid_data.pkl'
    try:
        with open(path_to_pkl, 'rb') as f:
            all_u = pickle.load(f)
            all_seqs = pickle.load(f)
            all_time_matrix = pickle.load(f)
            all_time_matrix_origin = pickle.load(f)
            all_distance_matrix = pickle.load(f)
            all_distance_matrix_lat = pickle.load(f)
            all_distance_matrix_lon = pickle.load(f)
            all_item_matrix = pickle.load(f)  #
            all_labels = pickle.load(f)
            all_origin_seqs = pickle.load(f)
            all_origin_time_seqs = pickle.load(f)
            all_origin_dis_seqs = pickle.load(f)
    except Exception as e:
        print('valid exception： ', e)
        print('begin writing valid')
        print('path_to_pkl = ', path_to_pkl)
        all_u = []
        all_seqs = []
        all_time_matrix = []
        all_time_matrix_origin = []
        all_distance_matrix = []
        all_distance_matrix_lat = []
        all_distance_matrix_lon = []
        all_item_matrix = []
        all_labels = []
        all_origin_seqs = []
        all_origin_time_seqs = []
        all_origin_dis_seqs = []
        # for u in users:
        for u in tqdm(users, total=len(users), ncols=120,
                      leave=False, unit='b', desc='Generating valid dataset'):
            if len(train[u]) < 1 or len(valid[u]) < 1: continue
            if u % 1000 == 0:
                print('.', end='')
                sys.stdout.flush()

            for j in range(len(valid[u][0])):
                seq = np.zeros([args.maxlen], dtype=np.int32)
                time_seq = np.zeros([args.maxlen], dtype=np.int32)
                dis_seq = ['0,0'] * args.maxlen

                origin_seq = np.zeros([max_origin_seq_len], dtype=np.int32)
                origin_time_seq = np.zeros([max_origin_seq_len], dtype=np.int32)
                origin_dis_seq = ['0,0'] * max_origin_seq_len

                idx = args.maxlen - 1
                origin_idx = max_origin_seq_len - 1

                extra_seq = valid[u][0][:j]
                all_seq = train[u] + extra_seq

                for i in reversed(all_seq):
                    origin_seq[origin_idx] = i[0]
                    origin_time_seq[origin_idx] = i[1]
                    origin_dis_seq[origin_idx] = i[2]
                    origin_idx -= 1
                    if origin_idx == -1:
                        break
                for i in reversed(all_seq):
                    seq[idx] = i[0]
                    time_seq[idx] = i[1]
                    dis_seq[idx] = i[2]
                    idx -= 1
                    if idx == args.maxlen - args.maxlen_origin - 1: break

                rated = set(map(lambda x: x[0], all_seq))
                if valid[u][0][j][0] in rated:
                    rated.remove(valid[u][0][j][0])
                item_idx = list(rated)
                label = valid[u][0][j][0]

                time_matrix = computeRePos(time_seq, args.time_span)
                time_matrix_origin = computeRePos(origin_time_seq, args.time_span)
                dis_matrix, dis_matrix_lat, dis_matrix_lon = computedisPos(dis_seq, args.time_span)
                all_u.append(u)
                all_seqs.append(seq)
                all_time_matrix.append(time_matrix)
                all_time_matrix_origin.append(time_matrix_origin)
                all_distance_matrix.append(dis_matrix)
                all_distance_matrix_lat.append(dis_matrix_lat)
                all_distance_matrix_lon.append(dis_matrix_lon)
                all_item_matrix.append(item_idx)
                all_labels.append(label)
                all_origin_seqs.append(origin_seq)
                all_origin_time_seqs.append(origin_time_seq)
                all_origin_dis_seqs.append(origin_dis_seq)

        with open(path_to_pkl, 'wb') as f:
            pickle.dump(all_u, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_seqs, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_time_matrix, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_time_matrix_origin, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_distance_matrix, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_distance_matrix_lat, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_distance_matrix_lon, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_item_matrix, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_labels, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_origin_seqs, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_origin_time_seqs, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_origin_dis_seqs, f, pickle.HIGHEST_PROTOCOL)
    for i in tqdm(range(len(users) // args.test_batch_size), total=len(users) // args.test_batch_size, ncols=70,
                  leave=False, unit='b', desc='Validing...'):
        if i < len(users) // args.test_batch_size:
            u = all_u[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            seq = all_seqs[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            time_matrix = all_time_matrix[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            time_matrix_origin = all_time_matrix_origin[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            dis_matrix = all_distance_matrix[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            dis_matrix_lat = all_distance_matrix_lat[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            dis_matrix_lon = all_distance_matrix_lon[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            item_idx = all_item_matrix[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            label = all_labels[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            origin_seq = all_origin_seqs[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            origin_time_seq = all_origin_time_seqs[i * args.test_batch_size:(i + 1) * args.test_batch_size]
            origin_dis_seq = all_origin_dis_seqs[i * args.test_batch_size:(i + 1) * args.test_batch_size]
        else:
            u = all_u[i * args.test_batch_size:]
            seq = all_seqs[i * args.test_batch_size:]
            time_matrix = all_time_matrix[i * args.test_batch_size:]
            time_matrix_origin = all_time_matrix_origin[i * args.test_batch_size:]
            dis_matrix = all_distance_matrix[i * args.test_batch_size:]
            dis_matrix_lat = all_distance_matrix_lat[i * args.test_batch_size:]
            dis_matrix_lon = all_distance_matrix_lon[i * args.test_batch_size:]
            item_idx = all_item_matrix[i * args.test_batch_size:]
            label = all_labels[i * args.test_batch_size:]
            origin_seq = all_origin_seqs[i * args.test_batch_size:]
            origin_time_seq = all_origin_time_seqs[i * args.test_batch_size:]
            origin_dis_seq = all_origin_dis_seqs[i * args.test_batch_size:]

        predictions = -model.predict(sess=sess, args=args, u=u, seq=seq, time_matrix=time_matrix,
                                     time_matrix_origin=time_matrix_origin, dis_matrix=dis_matrix,
                                     dis_matrix_lat=dis_matrix_lat, dis_matrix_lon=dis_matrix_lon,
                                     item_idx=item_idx, adj_matrix=adj_matrix, tra_matrix=tra_matrix,
                                     origin_seq=origin_seq, origin_time_seq=origin_time_seq,
                                     origin_seq_location=origin_dis_seq, max_origin_seq_len=max_origin_seq_len,
                                     all_poi_dis_matrix=all_poi_dis_matrix,
                                     all_poi_dis_matrix_lat=all_poi_dis_matrix_lat,
                                     all_poi_dis_matrix_lon=all_poi_dis_matrix_lon, diagonal_zero=diagonal_zero,
                                     global_=i)
        mask_matrix = np.ones_like(predictions)
        for i in range(mask_matrix.shape[0]):
            for j in range(len(item_idx[i])):
                if item_idx[i][j] != label[i]:
                    mask_matrix[i, item_idx[i][j]] = 0

        predictions = predictions * mask_matrix
        ranks = predictions.argsort().argsort()
        rank = []
        for i in range(len(ranks)):
            rank.append(ranks[i, label[i]])

        valid_user += len(rank)
        for i in rank:
            if i < 2:
                NDCG[0] += 1 / np.log2(i + 2)
                HT[0] += 1
            if i < 5:
                NDCG[1] += 1 / np.log2(i + 2)
                HT[1] += 1
            if i < 10:
                NDCG[2] += 1 / np.log2(i + 2)
                HT[2] += 1

    return list(map(lambda x: x / valid_user, NDCG)), list(map(lambda x: x / valid_user, HT))


def get_adj_matrix(matrix):
    row_sum = np.array(matrix.sum(1)) + 1e-24
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(
        matrix.dot(degree_mat_inv_sqrt)).tocoo()
    # return rel_matrix_normalized
    indices = np.vstack((rel_matrix_normalized.row, rel_matrix_normalized.col)).transpose()
    # return indices
    values = rel_matrix_normalized.data.astype(np.float32)
    shape = rel_matrix_normalized.shape
    return indices, values, shape


def concat_similar_poi(new_seq, old_seq, origin_seq, old_dis_matrix, old_dis_matrix_lat, old_dis_matrix_lon,
                       old_time_matrix, old_time_matrix_origin,
                       all_poi_dis_matrix, all_poi_dis_matrix_lat, all_poi_dis_matrix_lon, args, similar_poi_idx_dict):
    longtime_time_matrix = old_time_matrix
    longtime_dis_matrix = old_dis_matrix
    longtime_dis_matrix_lat = old_dis_matrix_lat
    longtime_dis_matrix_lon = old_dis_matrix_lon
    for b in range(len(old_seq)):
        seq = old_seq[b].tolist()
        old_zero_num = seq.count(0)
        similar_poi_idx = similar_poi_idx_dict[b]
        if old_zero_num != args.max_longtime_avg_len or len(similar_poi_idx) == 0:
            continue
        old_seq[b][: args.max_longtime_avg_len] = new_seq[b][: args.max_longtime_avg_len]

        concated_seq_ = old_seq[b].tolist()
        similar_poi_idx = similar_poi_idx[::-1]
        for i in range(len(similar_poi_idx)):
            t = similar_poi_idx[i]
            insert_idx = args.max_longtime_avg_len - len(similar_poi_idx) + i
            longtime_time_matrix[b][insert_idx, -args.maxlen_origin:] = old_time_matrix_origin[b][t, -args.maxlen_origin:]
            longtime_time_matrix[b][-args.maxlen_origin:, insert_idx] = old_time_matrix_origin[b][-args.maxlen_origin:, t]
            for j in range(len(similar_poi_idx)):
                tt = similar_poi_idx[j]
                insert_idx_j = args.max_longtime_avg_len - len(similar_poi_idx) + j
                longtime_time_matrix[b][insert_idx, insert_idx_j] = old_time_matrix_origin[b][t, tt]
                longtime_time_matrix[b][insert_idx_j, insert_idx] = old_time_matrix_origin[b][tt, t]

        for i in range(args.max_longtime_avg_len):
            for j in range(args.maxlen):
                longtime_dis_matrix[b][i][j] = all_poi_dis_matrix[concated_seq_[i]][concated_seq_[j]]
                longtime_dis_matrix_lat[b][i][j] = all_poi_dis_matrix_lat[concated_seq_[i]][concated_seq_[j]]
                longtime_dis_matrix_lon[b][i][j] = all_poi_dis_matrix_lon[concated_seq_[i]][concated_seq_[j]]

                longtime_dis_matrix[b][j][i] = all_poi_dis_matrix[concated_seq_[j]][concated_seq_[i]]
                longtime_dis_matrix_lat[b][j][i] = all_poi_dis_matrix_lat[concated_seq_[j]][concated_seq_[i]]
                longtime_dis_matrix_lon[b][j][i] = all_poi_dis_matrix_lon[concated_seq_[j]][concated_seq_[i]]
    return old_seq, longtime_dis_matrix, longtime_dis_matrix_lat, longtime_dis_matrix_lon, longtime_time_matrix

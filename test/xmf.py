from numpy import *

from rec_lib.evaluate import precision, recall
from rec_lib.heap import ZPriorityQ, KVTtem
from rec_lib.utils import read_checks_table, read_obj, write_obj, sort_dict, out_json_to_file
import os


class MyMFModel:
    def __init__(self, checks, K, dirname):
        self.K = K
        p_name = dirname + 'p.txt'
        q_name = dirname + 'q.txt'
        user_list_name = dirname + 'user-list.txt'
        item_list_name = dirname + 'item-list.txt'

        if os.path.exists(p_name) and os.path.exists(q_name) and os.path.exists(user_list_name) and  os.path.exists(item_list_name):
            self.p, self.q, self.users, self.items = read_obj(p_name), \
                                                     read_obj(q_name), \
                                                     read_obj(user_list_name), \
                                                     read_obj(item_list_name)
            self.M = len(self.users)
            self.N = len(self.items)
            print(self.M, self.N)
            self.user_index = {self.users[u]: u for u in range(len(self.users))}
            self.item_index = {self.items[i]: i for i in range(len(self.items))}
            self.R = {}
            for ou in checks.keys():
                u = self.user_index[ou]
                if not self.R.__contains__(u):
                    self.R[u] = {}
                for check in checks[u]:
                    i = self.item_index[check[0]]
                    if not self.R[u].__contains__(i):
                        self.R[u][i] = 0
                    self.R[u][i] = 1
        else:
            print('init user, items index')
            users = set()
            items = set()
            for u in checks.keys():
                users.add(u)
                for check in checks[u]:
                    i = check[0]
                    items.add(i)
            self.users = list(users)
            self.items = list(items)
            self.M = len(self.users)
            self.N = len(self.items)
            print(self.M, self.N)
            self.user_index = {self.users[u]: u for u in range(len(self.users))}
            self.item_index = {self.items[i]: i for i in range(len(self.items))}
            self.R = {}
            for ou in checks.keys():
                u = self.user_index[ou]
                if not self.R.__contains__(u):
                    self.R[u] = {}
                for check in checks[u]:
                    i = self.item_index[check[0]]
                    if not self.R[u].__contains__(i):
                        self.R[u][i] = 0
                    self.R[u][i] += 1
            # 初始化 参数 列表 成矩阵
            self.p, self.q = self.gradAscent()
            write_obj(p_name, self.p)
            write_obj(q_name, self.q)
            write_obj(user_list_name, self.users)
            write_obj(item_list_name, self.items)

    def gradAscent(self):
        print('gradAscent')
        p = random.random((self.M, self.K))
        q = random.random((self.N, self.K))
        alpha = 0.001
        beta = 0.02
        maxCycles = 1000
        pre = -1
        for step in range(maxCycles):
            # print(step)
            alpha *= 0.9
            beta *= 0.9
            tp = p.copy()
            tq = q.copy()
            for u in self.R.keys():
                for i in self.R[u].keys():
                    score = self.R[u][i]
                    error = score - dot(tp[u], tq[i].T)
                    p[u] += alpha * (2 * error * tq[i] - beta * tp[u])
                    q[i] += alpha * (2 * error * tp[u] - beta * tq[i])
            loss = 0.0
            for u in self.R.keys():
                for i in self.R[u].keys():
                    score = self.R[u][i]
                    error = dot(p[u], q[i])
                    # if error > 1:
                    #     print(p[u], q[i])
                    loss = (score - error) * (score - error)
                    loss += beta * (dot(p[u], p[u]).sum() + dot(q[i], q[i]).sum()) / 2
            if loss < 0.00002 : #or (pre > 0 and pre-loss < 0.0005)
                break
            if step % 10 == 0:
                print(step, loss, loss-pre)
                for i in self.R.get(0).keys():
                    print(self.R[0][i], dot(p[0], q[i]))
            pre = loss

        return p, q

    def predict(self, u, i):
        return dot(self.p[u], self.q[i])

    def recommend(self):
        rec = {}
        print('calculate predict')
        r = dot(self.p, self.q.T)
        print('generate recommendation list')
        for u in range(len(r)):
            if u % 10 == 0:
                print(u)
            rec_list = []
            for i in range(len(r[u])):
                if not self.R.get(u, {}).get(i, 0):
                    rec_list.append((self.items[i], r[u][i]))
                else:
                    if u == 0:
                        print(r[u][i], self.R[u][i])
            rec_list.sort(key=lambda d: d[1], reverse=True)
            rec[self.users[u]] = rec_list[: 20]
        return rec


def main(train_file, test_file, feature_num, topks):

    if not os.path.exists('mid_data/' + '-'.join(train_file.split('.')[:-1]) + '/'):
        os.makedirs('mid_data/' + '-'.join(train_file.split('.')[:-1]) + '/')
    nprs = []
    nres = []
    print('read_table')
    checks = read_checks_table(train_file, uin=0, iin=1)
    test = read_checks_table(test_file, uin=0, iin=1)

    sim_fun_name = 'pmf' + str(feature_num) + 't/'

    dir_name = 'mid_data/' + '-'.join(train_file.split('.')[:-1]) + '/' + sim_fun_name

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    mf_model = MyMFModel(checks, K=feature_num, dirname=dir_name)

    # for topn in topns:
    ex_rec_name = dir_name + '-'.join(['ex_rec']) + '.txt'
    if os.path.exists(ex_rec_name):
        print('read recommend result from file')
        rec = read_obj(ex_rec_name)
        for k, v in rec.items():
            v.reverse()
    else:
        print('recommend')
        rec = mf_model.recommend()
        write_obj(ex_rec_name, rec)
    prs = []
    res = []
    for topk in topks:
        pr = precision(rec, test, topk)
        print(pr)
        re = recall(rec, test, topk)
        prs.append(float('%.4f' % pr))
        res.append(float('%.4f' % re))
    nprs.append(prs.copy())
    nres.append(res.copy())
    out_json_to_file(dir_name + 'pr.txt', nprs)
    out_json_to_file(dir_name + 're.txt', nres)
    return nprs, nres


if __name__ == "__main__":
    train_file = 'trainid-id-dataset_TSMC2014_NYC.txt'
    test_file = 'testid-id-dataset_TSMC2014_NYC.txt'

    main(train_file, test_file, 5, [5, 10, 15, 20])

# if __name__ == '__main__':
#     checks = {0: [(0, 5), (1, 3), (3, 1)],
#               1: [(0, 4), (3, 1)],
#               2: [(0, 1), (1, 1), (3, 5)],
#               3: [(0, 1), (3, 4)],
#               4: [(1, 1), (2, 5), (3, 4)]}
#     mf = MyMFModel(checks, 2, 'test/')
#     print(mf.R)
#     print(dot(mf.p, mf.q.T))
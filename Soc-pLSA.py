# -*- coding:utf-8 -*-
import multiprocessing
import os
from datetime import datetime
from math import log
from random import random

import numpy as np

from rec_lib.evaluate import *
from rec_lib.heap import ZPriorityQ, KVTtem
from rec_lib.utils import read_checks_table, dic_value_reg_one, read_obj, write_obj, sort_dict, out_json_to_file, \
    read_dic_set


class MyLDA:
    @property
    def name(self):
        return 'soc-plsa'

    def __init__(self, train_filename, friends_file, topic_num, split_sig, time_format, uin, iin, timein, read_from_file=False):
        self.train_filename = train_filename
        self.topic_num = topic_num
        self.checks = read_checks_table(train_filename, split_sig=split_sig, time_format=time_format, uin=uin, iin=iin,
                                        timein=timein)
        self.friends = read_dic_set(friends_file, split_tag=split_sig, oin=0, ain=1)

        self.check_set = read_dic_set(train_filename, split_tag=split_sig, oin=0, ain=4)

        dir_name = 'mid_data/' + '-'.join(train_filename.split('.')[:-1]) + '/' + self.name + str(topic_num) + 't/'
        f_in_z_filename = dir_name + 'pr_f_in_z.txt'
        i_in_z_filename = dir_name + 'pr_i_in_z.txt'
        u_in_f_filename = dir_name + 'pr_u_in_f.txt'
        z_filename = dir_name + 'pz.txt'
        pr_filename = dir_name + 'pr.txt'

        max_user = max(self.check_set.keys())
        for i in range(max_user + 1):
            if self.friends.__contains__(i):
                self.friends[i].add(i)
            else:
                self.friends[i] = {i}

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        if os.path.exists(f_in_z_filename) and os.path.exists(i_in_z_filename) \
                and os.path.exists(z_filename) and os.path.exists(pr_filename) \
                and os.path.exists(u_in_f_filename):
            self.pr_f_in_z, \
            self.pr_i_in_z, \
            self.pr_u_in_f, \
            self.pz = read_obj(f_in_z_filename), \
                      read_obj(i_in_z_filename), \
                      read_obj(u_in_f_filename),\
                      read_obj(z_filename)
        else:
            self.pr_f_in_z, self.pr_i_in_z, self.pr_u_in_f, self.pz, self.pr = self.init_data()
            self.em_loop()
            self.pr_z_in_u = {}
            # for u in self.checks.keys():
            #     self.pr_z_in_u[u] = {}
            #     for z in range(len(self.pz)):
            #         self.pr_z_in_u[u][z] = 0
            #         for check in self.checks[u]:
            #             i = check[0]
            #             self.pr_z_in_u[u][z] += self.pr[(u, i)][z]
            #     dic_value_reg_one(self.pr_z_in_u[u])
            write_obj(f_in_z_filename, self.pr_f_in_z)
            write_obj(i_in_z_filename, self.pr_i_in_z)
            write_obj(z_filename, self.pz)
            write_obj(pr_filename, self.pr)
            write_obj(u_in_f_filename, self.pr_u_in_f)
        print(self.pz)

    def init_data(self):
        # 统计
        print('init_data')
        max_user = 0
        max_item = 0
        u_i_pairs = set()

        for u, u_checks in self.checks.items():
            max_user = int(u) if int(u) > max_user else max_user
            for check in u_checks:
                i = int(check[0])
                max_item = i if i > max_item else max_item
                u_i_pair = (u, i)
                u_i_pairs.add(u_i_pair)

        # 初始化参数 随机数
        pr_f_in_z = np.zeros((self.topic_num, max_user + 1))
        pr_i_in_z = np.zeros((self.topic_num, max_item + 1))
        pr_u_in_f = np.zeros((max_user + 1, max_user + 1))
        pz = np.zeros(self.topic_num)
        pr = {}

        for z in range(self.topic_num):
            pz[z] = random()
            for i in range(max_item + 1):
                pr_i_in_z[z][i] = random()
            for f in range(max_user+1):
                pr_f_in_z[z][f] = random()
            pr_i_in_z[z] /= pr_i_in_z[z].sum()
        pz /= pz.sum()

        for u, fs in self.friends.items():
            for f in fs:
                pr_u_in_f[f][u] = random()
        for f in range(len(pr_u_in_f)):
            # if pr_u_in_f[f].sum() < 0.0000001:
            #     print(f, pr_u_in_f[f])
            pr_u_in_f[f] /= pr_u_in_f[f].sum()

        for pair in u_i_pairs:
            pr[pair] = {}
            u, i = pair
            for z in range(self.topic_num):
                if not self.friends.get(u):
                    print(u)
                    print(self.friends)
                for f in self.friends.get(u):
                    zf = (z, f)
                    pr[pair][zf] = 1/(self.topic_num * len(self.friends.get(u)))
            # for zf in pr[pair].keys():
            #     pr[pair][zf] /= sum(pr[pair].values())

        return pr_f_in_z, pr_i_in_z, pr_u_in_f, pz, pr

    def e_step(self):
        print('E')
        # c = 0
        for pair in self.pr.keys():
            # if c % 1000 == 0:
            #     print(c, len(self.pr))
            # c+=1
            u, i = pair
            for f in self.friends[u]:
                for z in range(self.topic_num):
                    self.pr[pair][(z, f)] = self.pz[z] * self.pr_f_in_z[z, f] * self.pr_i_in_z[z, i] * self.pr_u_in_f[f][u]
            # try:
            #     if sum(self.pr[pair].values()) == 0:
            #         print(self.pr[pair])
            # except:
            #     print(self.pr[pair])
            for zf in self.pr[pair].keys():
                self.pr[pair][zf] /= sum(self.pr[pair].values())

    def m_step(self):
        print('M')

        # Pr(z)===================================
        for z in range(len(self.pz)):
            self.pz[z] = 0
            for pair in self.pr.keys():
                u, i = pair
                for f in self.friends.get(u):
                    self.pz[z] += self.pr[pair][(z, f)]
        self.pz /= self.pz.sum()

        # pr(f|z)====================================
        self.pr_f_in_z = np.zeros((len(self.pz), len(self.pr_f_in_z[0])))
        # for z in range(len(self.pz)):
        #     for f in range(len(self.pr_f_in_z[z])):
        #         self.pr_f_in_z[z][f] = 0

        for pair in self.pr.keys():
            for z in range(len(self.pz)):
                u, i = pair
                for f in self.friends.get(u):
                    self.pr_f_in_z[z][f] += self.pr[pair][(z, f)]
        for z in range(len(self.pz)):
            self.pr_f_in_z[z] /= self.pr_f_in_z[z].sum()

        # pr(u|f)
        self.pr_u_in_f = np.zeros((len(self.pr_u_in_f), len(self.pr_u_in_f[0])))
        # for us in self.pr_u_in_f:
        #     for u in us.keys():
        #         us[u]=0
        # for z in range(len(self.pz)):
        #     for f in range(len(self.pr_f_in_z[z])):
        #         self.pr_f_in_z[z][f] = 0
        for pair in self.pr.keys():
            u, i = pair
            for z in range(len(self.pz)):
                for f in self.friends.get(u):
                    self.pr_u_in_f[f][u] += self.pr[pair][(z, f)]
        for f in range(len(self.pr_u_in_f)):
            self.pr_u_in_f[f] /= self.pr_u_in_f[f].sum()

        # Pr(i|z)=====================================
        self.pr_i_in_z = np.zeros((len(self.pz),len(self.pr_i_in_z[0])))
        # for z in range(len(self.pz)):
        #     for i in range(len(self.pr_i_in_z[z])):
        #         self.pr_i_in_z[z][i] = 0
        for pair in self.pr.keys():
            u, i = pair
            for f in self.friends.get(u):
                for z in range(len(self.pz)):
                    self.pr_i_in_z[z][i] += self.pr[pair][(z, f)]
        for z in range(len(self.pz)):
            self.pr_i_in_z[z] /= self.pr_i_in_z[z].sum()

    # def m_step_no_num(self):
    #     print('M')
    #     for z in range(len(self.pz)):
    #         self.pz[z] = 0
    #         for pair in self.pr.keys():
    #             self.pz[z] += self.pr[pair][z]
    #     self.pz /= self.pz.sum()
    #
    #     for z in range(len(self.pz)):
    #         for u in range(len(self.pr_u_in_z[z])):
    #             self.pr_u_in_z[z][u] = 0
    #             try:
    #                 for check in self.check_set.get(u):
    #                     i = check
    #                     self.pr_u_in_z[z][u] += self.pr[(u, i)][z]
    #             except Exception as e:
    #                 print(u)
    #                 raise e
    #
    #         self.pr_u_in_z[z] /= self.pr_u_in_z[z].sum()
    #
    #     for z in range(len(self.pz)):
    #         for i in range(len(self.pr_i_in_z[z])):
    #             self.pr_i_in_z[z][i] = 0
    #         for u in range(len(self.pr_u_in_z[z])):
    #             for check in self.check_set.get(u):
    #                 i = check
    #                 self.pr_i_in_z[z][i] += self.pr[(u, i)][z]
    #         self.pr_i_in_z[z] /= self.pr_i_in_z[z].sum()

    def l(self):
        sumv = .0
        for u in self.checks.keys():
            for i in self.check_set[u]:
                temp = .0
                for f in self.friends.get(u):
                    for z in range(self.topic_num):
                        temp += self.pz[z] * self.pr_f_in_z[z][f] * self. pr_u_in_f[f][u] * self.pr_i_in_z[z][i]
                try:
                    sumv += log(temp)
                except:
                    pass
        return sumv

    def em_loop(self, max_loop=1000, stop_delta=1):
        count = 0
        pre_l = 0
        while count < max_loop:
            print(count)
            self.e_step()
            self.m_step()
            # self.m_step_no_num()
            l = self.l()
            if l - pre_l < stop_delta and count > 5:
                return
            pre_l = l
            print(l)
            print(self.pz)
            count += 1

    def predict(self, u, i):
        temp = .0
        try:
            for f in self.friends.get(u):
                for z in range(self.topic_num):
                    temp += self.pz[z] * self.pr_f_in_z[z][f] * self.pr_u_in_f[f][u] * self.pr_i_in_z[z][i]
        except TypeError as e:
            print(type(u), u)
            raise e
        return temp

        # def sim(self, u1, u2):
        #     return cosine_for_dic(self.pr_z_in_u[u1], self.pr_z_in_u[u2])


# def recommend(checks, sim_mat, topn, predict_fun):
#     rec = {}
#     for u in sim_mat:
#         count = 0
#         rec[u] = {}
#         for uss in sim_mat[u]:
#             count += 1
#             if count >= topn:
#                 break
#             for (item, score, time) in checks[uss[0]]:
#                 if not rec[u].keys().__contains__(item):
#                     rec[u][item] = predict_fun(u, item)
#
#     for user in rec.keys():
#         rec[user] = sort_dict(rec[user])
#     # print(rec[0])
#     return rec

def exclude_recommend(checks, users, locs, predic_fun):
    rec = {}
    c = 0
    for u in users:
        old_items = set([int(i[0]) for i in checks.get(u, [])])
        rec[u] = {}
        c += 1
        for l in locs:
            if not old_items.__contains__(l):
                rec[u][l] = predic_fun(int(u), int(l))
        del old_items
        if c % 100 == 0:
            print(c / 100)
        rec[u] = sort_dict(rec[u])[:100]
    return rec


# main
def cf_main(train_file, test_file, friend_file, topns=None, topks=None, topic_num=8):
    start = datetime.now()
    if topks is None:
        topks = [20]
    if topns is None:
        topns = [20]
    nprs = []
    nres = []

    print('read_table')
    table = read_checks_table(train_file, split_sig='\t', uin=0, iin=4, timein=1, scorein=None,
                              time_format='%Y-%m-%dT%H:%M:%SZ')
    test = read_checks_table(test_file, split_sig='\t', uin=0, iin=4, timein=1, scorein=None,
                             time_format='%Y-%m-%dT%H:%M:%SZ')

    # table = read_checks_table(train_file, split_sig=',', uin=0, iin=4, timein=3, scorein=None,
    #                           time_format='%Y-%m-%d %H:%M:%S')
    # test = read_checks_table(test_file, split_sig=',', uin=0, iin=4, timein=3, scorein=None,
    #                          time_format='%Y-%m-%d %H:%M:%S')

    # friends_dic = read_dic_set('Gowalla_edges.txt')

    if not os.path.exists('mid_data/' + '-'.join(train_file.split('.')[:-1]) + '/'):
        os.mkdir('mid_data/' + '-'.join(train_file.split('.')[:-1]) + '/')

    # ========= LDA ================

    lda = MyLDA(train_filename=train_file, friends_file=friend_file, topic_num=topic_num, split_sig='\t', uin=0, iin=4, timein=1,
                time_format='%Y-%m-%dT%H:%M:%SZ')
    # lda = MyLDA(train_filename=train_file, topic_num=topic_num, split_sig=',', uin=0, iin=4, timein=3, time_format='%Y-%m-%d %H:%M:%S')

    # sim_fun = lambda u1, u2: lda.sim(u1, u2)
    predict_fun = lda.predict
    # '''
    sim_fun_name = lda.name + str(topic_num) + 't'
    dir_name = 'mid_data/' + '-'.join(train_file.split('.')[:-1]) + '/' + sim_fun_name + '/'
    sim_name = dir_name + 'sim.txt'

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # if os.path.exists(sim_name):
    #     print('read sim metrics from file')
    #     sim_metrics = read_obj(sim_name)
    # else:
    #     print('cal_sim_mat')
    #     sim_metrics = cal_sim_mat(table, similar_fun=sim_fun)
    #     write_obj(sim_name, sim_metrics)

    for topn in topns:
        ex_rec_name = dir_name + '-'.join(['ex_rec', sim_fun_name, str(topn)]) + '.txt'
        if os.path.exists(ex_rec_name):
            print('read recommend result from file')
            rec = read_obj(ex_rec_name)
        else:
            print('recommend')
            users = set(table.keys())
            items = set()
            zp = ZPriorityQ(maxsize=1000)
            for z in range(len(lda.pr_i_in_z)):
                for i in range(len(lda.pr_i_in_z[z, :])):
                    zp.enQ(KVTtem(i, lda.pr_i_in_z[z, i]))
                items.update([e.k for e in zp.items])
            print(len(items))
            # for item, v in lda.pr_i_in_z[0].items():
            #     items.add(item)
            rec = exclude_recommend(table, users, items, predict_fun)
            # write_obj(rec_name, rec)
            # exclude_dup(table, rec)
            write_obj(ex_rec_name, rec)

        prs = []
        res = []
        for topk in topks:
            print('precision')
            pr = precision(rec, test, topk)
            print(pr)
            re = recall(rec, test, topk)
            print('recall')
            prs.append(float('%.4f' % pr))
            res.append(float('%.4f' % re))
        # print('y1=',prs)
        # print('y2=',res)
        nprs.append(prs.copy())
        nres.append(res.copy())
    out_json_to_file(dir_name + 'nprs.txt', nprs)
    out_json_to_file(dir_name + 'nres.txt', nres)

    end = datetime.now()
    print('the cost time is ', (end - start).seconds)

    return nprs, nres


if __name__ == '__main__':
    # train_file = 'trainRF-SH-FoursquareCheckins.csv'
    # test_file = 'testRF-SH-FoursquareCheckins.csv'
    # train_file = 'trainid-id-RF-NA-Gowalla_totalCheckins.txt'
    # test_file = 'testid-id-RF-NA-Gowalla_totalCheckins.txt'
    # friend_file = 'id-id-RF-NA-Gowalla_edges.txt'
    train_file = 'trainid-id-NYS-Gowalla_totalCheckins.txt'
    test_file = 'testid-id-NYS-Gowalla_totalCheckins.txt'
    friend_file = 'id-id-NYS-Gowalla_edges.txt'

    pool = multiprocessing.Pool(processes=2)
    results = []
    for z in [10, 12]:
        results.append(pool.apply_async(func=cf_main, args=(train_file,
                                                            test_file,
                                                            friend_file,
                                                            [5],
                                                            [5, 10, 15, 20],
                                                            z)))
    pool.close()
    pool.join()
    nprs = []
    nres = []
    for result in results:
        nprs.append(result.get()[0][0])
        nres.append(result.get()[1][0])
    pprint(nprs)
    pprint(nres)

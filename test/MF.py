#!/usr/bin/python
# coding=utf-8
from pprint import pprint

from rec_lib.evaluate import *
from rec_lib.similar import *
from rec_lib.utils import *
from rec_lib.xmf import MyMFModel
import os


def main(train_file, test_file, feature_num, topns=[20], topks=[20]):

    if not os.path.exists('mid_data/' + '-'.join(train_file.split('.')[:-1]) + '/'):
        os.makedirs('mid_data/' + '-'.join(train_file.split('.')[:-1]) + '/')
    nprs = []
    nres = []
    print('read_table')
    checks = read_checks_table(train_file,uin=0,iin=4,timein=1,scorein=None)
    test = read_checks_table(test_file, uin=0,iin=4,timein=1,scorein=None)
    # friends_dic = read_dic_set('Gowalla_edges.txt')

    sim_fun_name = 'mf' + str(feature_num) + 't/'

    dir_name = 'mid_data/' + '-'.join(train_file.split('.')[:-1]) + '/' + sim_fun_name
    mf_model = MyMFModel(checks, K=feature_num, dirname=dir_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    sim_name = dir_name + 'sim.txt'
    if os.path.exists(sim_name):
        sim_metrics = read_obj(sim_name)
    else:
        sim_metrics = cal_sim_mat(checks, mf_model.sim)
        write_obj(sim_name, sim_metrics)

    for topn in topns:
        rec_name = dir_name + '-'.join(['rec', str(topn)]) + '.txt'
        ex_rec_name = dir_name + '-'.join(['ex_rec', str(topn)]) + '.txt'
        if os.path.exists(ex_rec_name):
            print('read recommend result from file')
            rec = read_obj(ex_rec_name)
        else:
            print('recommend')
            rec = recommend(checks, sim_metrics, topn, mf_model.predict)
            write_obj(rec_name, rec)
            exclude_dup(checks, rec)
            write_obj(ex_rec_name, rec)

        prs = []
        res = []
        for topk in topks:
            # print('precision')
            pr = precision(rec, test, topk)
            print(pr)
            re = recall(rec, test, topk)
            # print('recall')
            # print(re)
            prs.append(float('%.4f' % pr))
            res.append(float('%.4f' % re))
        # print('y1=',prs)
        # print('y2=',res)
        nprs.append(prs.copy())
        nres.append(res.copy())
    return nprs,nres


# topn 邻居数
# recn 推荐数量
def recommend(op_table, sim_metrics, topn, predict_fun):
    rec = {}
    for u in sim_metrics:
        count = 0
        rec[u] = {}
        for uss in sim_metrics[u]:
            count += 1
            if count >= topn:
                break
            # for (item, score, time) in op_table[uss[0]]:
            #     if not rec[u].keys().__contains__(item):
            #         rec[u][item] = predict_fun(u, item)
            for (item, score, time) in op_table[uss[0]]:
                if rec[u].keys().__contains__(item):
                    rec[u][item] += uss[1] * score
                else:
                    rec[u][item] = uss[1] * score
    for user in rec.keys():
        rec[user] = sort_dict(rec[user])
    # print(rec['9232'])
    return rec          

def recommend_by_score(op_table, predict_fun):
    rec = {}
    for u in sim_metrics:
        count = 0
        rec[u] = {}
        for uss in sim_metrics[u]:
            count += 1
            if count >= topn:
                break
            for (item, score, time) in op_table[uss[0]]:
                if not rec[u].keys().__contains__(item):
                    rec[u][item] = predict_fun(u, item)
            # for (item, score, time) in op_table[uss[0]]:
            #     if rec[u].keys().__contains__(item):
            #         rec[u][item] += uss[1] * score
            #     else:
            #         rec[u][item] = uss[1] * score
    for user in rec.keys():
        rec[user] = sort_dict(rec[user])
    # print(rec['9232'])
    return rec                  

def validate_mf():
    train_file = 'trainklnd-Gowalla_totalCheckins.txt'
    checks =read_checks_table('trainklnd-Gowalla_totalCheckins.txt')
    sim_fun_name = 'mf' + str(10) + 't/'

    dir_name = 'mid_data/' + '-'.join(train_file.split('.')[:-1]) + '/' + sim_fun_name
    mf_model = MyMFModel(checks, K=10, dirname=dir_name)
    for u in checks.keys():
        for c in checks[u]:
            i = c[0]
            print(mf_model.predict(u, i))
        if random.random()<0.2:
            break


if __name__ == '__main__':
    nprs, nres = main(train_file='trainid-id-NY-Gowalla_totalCheckins.txt',
        feature_num=10,
        test_file='testid-id-NY-Gowalla_totalCheckins.txt',
        topns=[300],
        topks=[5,10,15,20])
    pprint(nprs)
    pprint(nres)
    # validate_mf()
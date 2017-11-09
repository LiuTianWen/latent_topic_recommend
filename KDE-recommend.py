from datetime import datetime
import os
from math import radians, cos, sin, asin, sqrt
import numpy as np

from center import read_user_center
from rec_lib.evaluate import precision, recall
from rec_lib.utils import sort_dict, read_checks_table, out_json_to_file, read_obj, write_obj


# 用于计算和存储 经纬度距离，经纬度全部省略到两位数
class CacheHaverSine:

    @staticmethod
    def co_hash(*args):
        ele = ["%.2f" % arg for arg in args]
        return ','.join(ele), tuple(map(float, ele))

    @property
    def name(self):
        return 'mid_data/'+'haversine.cache'

    def __init__(self):
        if os.path.exists(self.name):
            self.cache = read_obj(self.name)
        else:
            self.cache = {}
        self.size = len(self.cache)

    def put(self, key, value):
        if self.cache. __contains__(key):
            return
        self.cache[key] = value
        self.size += 1
        # if self.size % 1 == 0:
        #     write_obj(self.name, self.cache)

    def get(self, key):
        return self.cache.get(key)

    def haversine_dis(self, lat1, lon1, lat2, lon2):
        """ 
        Calculate the great circle distance between two points  
        on the earth (specified in decimal degrees) 
        """
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # 地球平均半径，单位为公里
        return c * r

    def __call__(self, lat1, lon1, lat2, lon2):
        if lat1 > lat2:
            lat1, lat2, lon1, lon2 = lat2, lat1, lon2, lon1
        hashcode, (lat1, lon1, lat2, lon2) = CacheHaverSine.co_hash(lat1, lon1, lat2, lon2)

        if self.cache.__contains__(hashcode):
            return self.cache.get(hashcode)
        else:
            result = self.haversine_dis(lat1, lon1, lat2, lon2)
            self.put(hashcode, result)
            return result

    def save(self):
        write_obj(self.name, self.cache)

# 用于计算和存储 高斯核函数，距离全部化为整数，需要参数 h
class CacheKernel:

    @property
    def name(self):
        return 'mid_data/'+str(self.h)+'.cache_kernel'

    def __init__(self, h):
        self.h = h
        if os.path.exists(self.name):
            self.cache = read_obj(self.name)
        else:
            self.cache = {}
        self.size = len(self.cache)

    def put(self, key, value):
        if self.cache. __contains__(key):
            return
        self.cache[key] = value
        self.size += 1
        # if self.size % 1 == 0:
        #     write_obj(self.name, self.cache)

    def get(self, key):
        return self.cache.get(key)

    def kernel(self, delta):
        return np.exp(- delta*delta / (2 * self.h * self.h))

    def __call__(self, delta):
        delta = int(delta)
        if self.cache.__contains__(delta):
            return self.cache.get(delta)
        else:
            result = self.kernel(delta)
            self.put(delta, result)
            return result

    def save(self):
        write_obj(self.name, self.cache)

haversine = CacheHaverSine()
kernel = CacheKernel(10)


def kde(centers, visited, loc):
    p = 0
    la, lo = centers[loc]
    for tloc in visited:
        tla, tlo = centers[tloc]
        p += kernel(haversine(la, lo, tla, tlo))
    return p


def recommend(locs, centers, visited):
    rec = {}
    for loc in locs - visited:
        rec[loc] = kde(centers, visited, loc)
    return sort_dict(rec)

class CacheKDERecommend:
    @property
    def name(self):
        return 'mid_data/'+str(self.dir_name)+'l_in_Ru.cache'

    def __init__(self, dir_name):
        self.dir_name = dir_name
        if os.path.exists(self.name):
            self.cache = read_obj(self.name)
        else:
            self.cache = {}
        self.size = len(self.cache)

    def put(self, key, value):
        if self.cache. __contains__(key):
            return
        self.cache[key] = value
        self.size += 1
        # if self.size % 1 == 0:
        #     write_obj(self.name, self.cache)

    def get(self, key):
        return self.cache.get(key)

    def kernel(self, delta):
        return np.exp(- delta*delta / (2 * self.h * self.h))

    def __call__(self, delta):
        delta = int(delta)
        if self.cache.__contains__(delta):
            return self.cache.get(delta)
        else:
            result = self.kernel(delta)
            self.put(delta, result)
            return result

    def save(self):
        write_obj(self.name, self.cache)

def kde_recommend(train_checks, centers):
    # Rus = {k: [[c[3], c[4]] for c in v] for k, v in train_checks.items()}
    uvisited = {k: set([c[0] for c in v]) for k, v in train_checks.items()}
    locs = set([int(k) for k in centers.keys()])
    rec = {}
    for u in train_checks.keys():
        start = datetime.now()
        rec[u] = recommend(locs, centers, uvisited[u])[:100]
        end = datetime.now()
        print(u, (end-start).seconds)
        if u % 10 == 0:
            haversine.save()
            kernel.save()
    return rec


if __name__ == '__main__':
    train_file = 'trainid-id-dataset_TSMC2014_NYC.txt'
    test_file = 'testid-id-dataset_TSMC2014_NYC.txt'
    center_file = 'loc-center-trainid-id-dataset_TSMC2014_NYC.txt'
    print('read_file now')
    train = read_checks_table(train_file, split_sig='\t', uin=0, iin=1, lain=4, loin=5)
    test = read_checks_table(test_file, split_sig='\t', uin=0, iin=1, lain=4, loin=5)
    centers = read_user_center(center_file)
    print('recommend_now')
    rec = kde_recommend(train, centers)
    topks = [5,10,15]
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
    print(prs, res)

    kernel.save()
    haversine.save()
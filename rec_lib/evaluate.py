# 计算准确率
# topk 推荐数量
from pprint import pprint


def precision(rec, test_table, topk):
    test = {}
    for k, v in test_table.items():
        test[k] = {e[0] for e in v}
    down = 0
    up = 0
    for k, v in rec.items():
        v = set([e[0] for e in v][:topk])
        down += len(v)
        up += len(v & test.get(k, set()))
    # print(up, down)
    return up / down


# 计算召回
# topk 推荐数量
def recall(rec, test_table, topk):
    test = {}
    down = 0
    for k, v in test_table.items():
        test[k] = {e[0] for e in v}
        down += len(test[k])

    up = 0
    for k, v in rec.items():
        v = set([e[0] for e in v][:topk])
        up += len(v & test.get(k, set()))
    # print(up, down)
    return up / down


def variety(rec, topk):
    r = set()
    for k, v in rec.items():
        r.update([e[0] for e in v][:topk])
    return len(r)


class D_Recall:

    def __init__(self, test_table, topks, topns):
        self.test_table = test_table
        self.down = 0
        self.topks = topks
        self.topns = topns
        self.max_top = max(self.topks)
        self.rc = {n: {k: 0 for k in topks} for n in topns}

    def add_recs(self, user, recs):
        user_test = set([e[0] for e in self.test_table.get(user, [])])
        for n, rec in recs.items():
            count = 0
            for e in rec:
                count += 1
                if count > self.max_top:
                    break
                i = e[0]
                if user_test.__contains__(i):
                    for k in self.rc[n].keys():
                        if k >= count:
                            self.rc[n][k] += 1
        # pprint(self.test.get(user, set()))
        self.down += len(user_test)

    def get_result(self):
        if self.down > 0:
            return {n: {k: self.rc[n][k]/self.down for k in self.topks} for n in self.topns}
        return None


class D_Precision:

    def __init__(self, test_table, topks, topns):
        self.test_table = test_table
        self.topks = topks
        self.topns = topns
        self.max_top = max(self.topks)
        self.rc = {n: {k: 0 for k in topks} for n in topns}
        self.afc = {n: {k: 0 for k in topks} for n in topns}

    def add_recs(self, user, recs):
        user_test = set([e[0] for e in self.test_table.get(user, [])])
        for n, rec in recs.items():
            count = 0
            for e in rec:
                count += 1
                if count > self.max_top:
                    break
                i = e[0]
                for k in self.afc[n].keys():
                    self.afc[n][k] += min(k, len(rec))
                if user_test.__contains__(i):
                    for k in self.rc[n].keys():
                        if k >= count:
                            self.rc[n][k] += 1

    def get_result(self):

        return {n: {k: 0 if self.afc[n][k]==0 else self.rc[n][k]/self.afc[n][k] for k in self.topks} for n in self.topns}

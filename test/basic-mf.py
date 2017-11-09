from numpy import random
import numpy as np


class BasicMF:

    def __init__(self, checks, M, N, K):

        self.M = M
        self.N = N
        self.K = K
        self.R = np.zeros((M, N))
        for u, uchecks in checks.items():
            for check in uchecks:
                i = check[0]
                self.R[u][i] = check[1]
        self.p, self.q = self.gradAscent()

    def gradAscent(self):
        p = random.random((self.M, self.K))
        q = random.random((self.N, self.K))
        # p = np.array([1.0 for i in range(self.M*self.K)]).reshape((self.M, self.K))
        # q = np.array([1.0 for i in range(self.N*self.K)]).reshape((self.N, self.K))
        alpha = 0.001
        beta = 0.02
        maxCycles = 2000
        pre = 100
        for step in range(maxCycles):
            # print(step)
            error = self.R - np.dot(p, q.T)
            for u in range(self.M):
                for i in range(self.N):
                    score = self.R[u][i]
                    if score == 0:
                        continue
                    # error = score - np.dot(p[u], q[i])
                    tp = p.copy()
                    tq = q.copy()
                    p[u] += alpha * (2 * error[u][i] * tq[i] - beta * tp[u])
                    q[i] += alpha * (2 * error[u][i] * tp[u] - beta * tq[i])
            if step == 0:
                print(p)
                print(q)
            loss = 0.0
            for u in range(self.M):
                for i in range(self.N):
                    score = self.R[u][i]
                    error = np.dot(p[u], q[i])
                    loss = (score - error) * (score - error)
                    loss += beta * (np.dot(p[u], p[u]).sum() + np.dot(q[i], q[i]).sum()) / 2
            if loss < 0.002:
                break
            pre = loss
            if step % 10 == 0:
                print(step, loss)
        return p, q

    def predict(self):
        return np.dot(self.p, self.q.T)


if __name__ == '__main__':
    checks = {0: [(0, 1), (1, 1), (3, 1)],
              1: [(0, 1), (3, 1)],
              2: [(0, 1), (1, 1), (3, 1)],
              3: [(0, 1), (3, 1)],
              4: [(1, 1), (2, 1), (3, 1)]}
    mf = BasicMF(checks, 5, 4, 2)
    print(mf.R)
    print(mf.predict())
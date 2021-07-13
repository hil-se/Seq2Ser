import numpy as np


class Seq2Ser:

    def __init__(self, k):
        self.k = k

    def transform(self, X):
        return np.array(self.trans(np.array(X), self.k)).flatten()/len(X)

    def fit_transform(self, X):
        return self.transform(X)

    def trans(self, X, k):
        y = []
        m = X.shape[0]
        if len(X.shape) > 1:
            n = X.shape[1]
        else:
            n = 1
        if k <= 0:
            return y
        if m == 0:
            s = np.array([0.0]*n)
        else:
            s = np.sum(X, axis = 0)
        XX = X - s / float(m)
        # y.append(m*(2**(k-1)))
        y.append(s)
        split = int(np.ceil(m/2.0))
        XXl = XX[:split]
        XXr = XX[split:]
        j = 0
        id = 0
        yl = self.trans(XXl, k-1)
        yr = self.trans(XXr, k-1)
        while j<k-1:
            y.extend(yl[id:id+2**j])
            y.extend(yr[id:id + 2 ** j])
            id+=2**j
            j+=1
        return y


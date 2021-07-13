import gensim
import pandas as pd
import numpy as np
from seq2ser import Seq2Ser
from pdb import set_trace

class DocEmbedding:

    def __init__(self):
        self.wv = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin', binary=True)
        self.vecSize = 300

    def loadData(self, path = "../data/quora.csv"):
        self.data = pd.read_csv(path).dropna()
        self.y = self.data["is_duplicate"].to_numpy()

    def preprocess(self):
        question1 = []
        for sentence in self.data["question1"]:
            question1.append(gensim.utils.simple_preprocess(sentence))
        question2 = []
        for sentence in self.data["question2"]:
            question2.append(gensim.utils.simple_preprocess(sentence))
        self.question1 = np.array(question1)
        self.question2 = np.array(question2)

    def embed(self, word):
        if word in self.wv:
            return list(self.wv[word])
        else:
            return [0.0] * self.vecSize

    def evaluate(self, order):
        fp = 0
        tp = 0.0
        auc = 0.0
        for label in self.y[order]:
            if label==1:
                tp += 1.0
            else:
                fp += 1.0
                auc += tp
        auc = auc / tp / fp
        return auc

    def test_average_pooling(self):
        dists = []
        for i in range(len(self.y)):
            vecs = []
            for word in self.question1[i]:
                vecs.append(self.embed(word))
            vecs = np.array(vecs)
            q1 = np.mean(vecs, axis=0)
            vecs = []
            for word in self.question2[i]:
                vecs.append(self.embed(word))
            vecs = np.array(vecs)
            q2 = np.mean(vecs, axis=0)
            dist = np.linalg.norm(q1 - q2)
            dists.append(dist)
        order = np.argsort(dists)
        auc = self.evaluate(order)
        print("AUC of average pooling: %f" %auc)

    def test_seq2ser(self, k = 2):
        seq2ser = Seq2Ser(k)
        dists = []
        for i in range(len(self.y)):
            vecs = []
            for word in self.question1[i]:
                vecs.append(self.embed(word))
            vecs = np.array(vecs)
            if len(vecs) == 0:
                q1 = [0.0] * ((2 ** k - 1) * self.vecSize)
            else:
                q1 = seq2ser.transform(vecs)
            vecs = []
            for word in self.question2[i]:
                vecs.append(self.embed(word))
            vecs = np.array(vecs)
            if len(vecs) == 0:
                q2 = [0.0] * ((2 ** k - 1) * self.vecSize)
            else:
                q2 = seq2ser.transform(vecs)
            dist = np.linalg.norm(q1 - q2)
            dists.append(dist)
        order = np.argsort(dists)
        auc = self.evaluate(order)
        print("AUC of Seq2Ser: %f" %auc)

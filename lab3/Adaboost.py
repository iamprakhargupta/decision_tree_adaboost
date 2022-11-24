"""
CSCI-630: Foundation of AI
Adaboost.py
Autor: Prakhar Gupta
Username: pg9349

Implementing adaboost

"""


import numpy as np
import pandas as pd
from DecisionTree import DecisionTree


class AdaBoost:

    def __init__(self, ensembletype="dtree", nboost=10):
        self.ensembles = []
        self.alphas = []
        self.ensembletype = ensembletype
        self.weakclassifier = None
        self.train_error = []
        self.pred_error = []
        self.nboost = nboost

    def get_weight_sample(self,X,y,weight):





        y=np.asarray([y])
        f=np.concatenate((X, y.T), axis=1)
        f=pd.DataFrame(f)

        f["weight"]=weight

        f=f.sample(n=len(X), replace=True, random_state=42,weights='weight')
        # print(f)
        data=f.to_numpy()
        y=data[:,-2]
        X=data[:,:-2]
        # new=np.random.choice(f, len(y), weight)
        # X=new[:,:-1]
        # y=new[:,-1]
        return X,y

    def get_error(self,y,pred,weight):
        error_counter = 0
        error_sum=0
        for i in range(len(pred)):
            if pred[i] != y[i]:
                error_counter += 1
                error_sum+=weight[i]
        errorp = error_sum
        return errorp

    def _train(self, X, y):
        if self.ensembletype == "dtree":
            for i in range(0, self.nboost):
                if i == 0:
                    weight = np.asarray([1 / len(y)] * len(y))
                else:
                    weight = self.update_weights(weight, alpha, pred, y)

                dt = DecisionTree(level=1)
                X,y=self.get_weight_sample(X,y,weight)
                dt.train(X,y)
                pred=dt.pred(X)
                self.ensembles.append(dt)

                error=self.get_error(y,pred,weight)
                self.train_error.append(error)
                if error==0:
                    error=0.0001

                alpha=np.log((1-error)/error)
                self.alphas.append(alpha)






    def update_weights(self, weight, alpha, pred, y):
        error_counter = 0
        error_sum=0
        for i in range(len(pred)):
            if pred[i] != y[i]:
                error_counter += 1
                error_sum+=weight[i]
        errorp = error_sum
        update = errorp / (1 - errorp)
        sumofweight = 0
        for i in range(len(pred)):
            if pred[i] == y[i]:
                weight[i] = weight[i] * update
            sumofweight += weight[i]

        return weight / sumofweight

    def train(self, data, labels):
        self._train(data, labels)

    def pred(self,X):
        test=[]
        for i in X:
            test.append(self._pred(i))
        return test
    def _pred(self,x):
        p=[]
        x=np.asarray([x])
        # print(x)
        for i in range(self.nboost):
            result=self.ensembles[i].pred(x)
            if result[0]=="en":
                p.append(1*self.alphas[i])
            else:
                p.append(-1*self.alphas[i])
        s=sum(p)
        if s<0:
            return "nl"
        else:
            return "en"






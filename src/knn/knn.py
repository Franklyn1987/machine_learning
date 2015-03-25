# coding=utf-8
# created by WangZhe on 2015/3/25
import numpy as np
from src.utils.data import *
from collections import defaultdict
import heapq
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier
class Knn(object):
    def __init__(self,k=1):
        self.k = k

    def fit(self,X,y):
        self.X = X.copy()
        self.y = y.copy()

    def cal_dis(self,x,y):
        return np.linalg.norm(x-y)
    def predict(self,X):
        X = check_arrays(X)
        predict_result = []
        for x in X:
            heap = []
            for data_x,data_y in zip(self.X,self.y):
                dis = self.cal_dis(data_x,x)

                if len(heap) < self.k:
                    heapq.heappush(heap,(-dis,data_y))
                else:
                    heapq.heappushpop(heap,(-dis,data_y))
            count= defaultdict(int)
            for dis,data_y in heap:
                count[data_y] += 1
            result = sorted(count.iteritems(),key=itemgetter(1),reverse=True)
            # print result
            predict_result.append(result[0][0])


        return np.array(predict_result).ravel()




if __name__ == "__main__":
    cls = Knn(5)
    x_train,y_train= read_data(r"G:\Program\python\machine learning\data\hw4_knn_train.dat")
    x_test, y_test = read_data(r"G:\Program\python\machine learning\data\hw4_knn_test.dat")

    # x_train = np.array([[2,3],[1,2]])
    # y_train = np.array([1,0])
    cls.fit(x_train,y_train)
    print 1 - score(x_train, y_train, cls.predict)
    print 1- score(x_test,y_test,cls.predict)


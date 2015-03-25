# coding=utf-8
# created by WangZhe on 2015/3/25
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from src.utils.data import *
class Kmeans(object):
    def __init__(self,k=1,iter=10):
        self.k = k
        self.iter = iter

    def cal_dis(self,x,y):
        # print 'x',x
        # print 'y',y
        dis = np.linalg.norm(x-y)
        # print 'dis',dis
        return dis

    def get_center(self,x,centers):
        min_dis = None
        min_center_index = -1
        for index, center in enumerate(centers):
            dis = self.cal_dis(center, x)
            # print index,dis
            if min_dis == None or dis < min_dis :
                min_dis = dis
                min_center_index = index
        # print 'min_index',min_center_index
        return min_center_index

    def fit(self,X):
        centers = []
        index_array = range(len(X))
        m,n = X.shape
        for i in random.sample(index_array,self.k):
            centers.append(X[i])


        for i in range(self.iter):
            centers_pois = []
            for i in range(self.k):
                centers_pois.append([])
            for row,x in enumerate(X):
                min_center_index = self.get_center(x,centers)
                # print 'min_index', min_center_index
                # print row,centers_pois
                centers_pois[min_center_index].append(row)

            # print centers
            # print centers_pois
            for i in range(self.k):
                centers[i] = np.average(X[centers_pois[i]],axis=0)
            # print centers

            # print centers
        self.centers = centers
        return centers

    def cal_e(self,X):
        error = 0
        for x in X:
            min_center_index = self.get_center(x,self.centers)
            dis = self.cal_dis(self.centers[min_center_index],x)
            error += dis **2
        return error/X.shape[0]











if __name__ == "__main__":
    x_train = read_data(r"G:\Program\python\machine learning\data\hw4_kmeans_train.dat",False)
    # x_train = np.array([[1,2],[3,4],[5,6]])
    cls = Kmeans(10,iter=500)
    print cls.fit(x_train)

    print cls.cal_e(x_train)


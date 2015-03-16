# coding=utf-8
# created by WangZhe on 2015/3/1
from src.tree.classifytree import ClassifyTree
from collections import Counter
import numpy as np
from src.utils.data import read_data,score
from sklearn import tree as a



class RandomForest():
    def __init__(self):
        pass

    def sample_data(self,X,y,max_features=1):
        if max_features > X.shape[1] :
            max_features = X.shape[1]
        idx = np.random.choice(X.shape[0],X.shape[0],replace=True)
        idy = np.random.choice(X.shape[1] ,max_features,replace=False)
        # print idx
        # print idy
        # new_x = X[idx,idy]
        # new_y = y[idx]
        # print idy
        return X[idx][:,idy],y[idx],idy


    def train(self,X,y,max_features=1,n_estimators=10):
        # self.trees = [ClassifyTree()] * n_estimators
        self.trees = []
        for i in xrange(n_estimators):
            sample_x,sample_y,idy = self.sample_data(X,y,max_features)
            # tree = a.DecisionTreeClassifier()
            tree = ClassifyTree()
            tree.fit(sample_x,sample_y)
            self.trees.append((tree,idy))
            # print idy




    def predict(self,X):
        result = [int(tree.predict(X[idy])) for tree,idy in self.trees]
        # print result
        count = Counter(result)
        # print count
        y,num = count.most_common(1)[0]
        return y

    def predict_prob(self,X,y):
        result = [tree.predict(X) for tree in self.trees]
        count = Counter(result)
        print count
        return count[y] * 1.0 /sum(count.values())



if __name__ == "__main__":
    x, y = read_data(u'G:\Program\python\coursera\机器学习技法\homework3\hw3_train.dat')

    # print x_matrix.shape, y_matrix.shape
    # Tree = ClassifyTree()
    # Tree.train(x,y)
    x_test, y_test = read_data(u'G:\Program\python\coursera\机器学习技法\homework3\hw3_test.dat')
    # print score(x_test, y_test, Tree.predict)

    for i in range(1,100):
        forest = RandomForest()
        forest.train(x,y,2,i)

        print i,score(x_test,y_test,forest.predict)
        # for tree in forest.trees[:5]:
        #     print score(x_test,y_test,tree.predict)


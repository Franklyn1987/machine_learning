# coding=utf-8
# created by WangZhe on 2015/3/14
from __future__ import division
import numpy as np
from warnings import warn
import copy
from src.utils.data import read_data,score
class AdaBoost(object):
    def __init__(self,classifier=None,max_iter=3):
        if not classifier:
            self.classifier = DecisionStump()
        self.max_iter = max_iter

    # def fit(self,X,y):
    #     if X.shape[0] < 2:
    #         raise ValueError('category need to be more than one')
    #     weights = 1.0 / X.shape[0] * np.ones(X.shape[0])
    #     self.alphas = []
    #     self.models = []
    #     for iter_num in range(self.max_iter):
    #         model = copy.deepcopy(self.classifier)
    #         model = model.fit(X,y,weights=weights)
    #         self.models.append(model)
    #         predict_array = model.predict(X)
    #         error = np.sum(weights[predict_array != y])
    #         if error == 0:
    #             self.alphas.append(1)
    #             print 'stop!!!!!!!!!!!'
    #             break
    #         alpha = float(0.5 * np.log((1.0-error)/max(error,1e-16)))
    #         print iter_num,error,alpha
    #         self.alphas.append(alpha)
    #         new_weights = weights* np.exp(-alpha * y * predict_array)
    #         # print weights
    #         weights = new_weights / np.sum(new_weights)
    #     self.alphas = np.array(self.alphas)


    def predict(self,X):
        predict_array = np.zeros(X.shape[0])
        for model,alpha in zip(self.models,self.alphas):
            predict_array += alpha * model.predict(X)
        return np.sign(predict_array)


    def fit(self, X, y):
        if X.shape[0] < 2:
            raise ValueError('category need to be more than one')
        weights = 1.0 / X.shape[0] * np.ones(X.shape[0])
        self.alphas = []
        self.models = []
        error = 0
        for iter_num in range(self.max_iter):
            # print iter_num,np.sum(weights),error
            model = copy.deepcopy(self.classifier)
            model = model.fit(X, y, weights=weights)
            self.models.append(model)
            predict_array = model.predict(X)

            error = np.sum(weights[predict_array != y])/np.sum(weights)
            t = ((1.0 - error) / error) ** 0.5

            alpha = np.log(t)
            self.alphas.append(alpha)
            weights[predict_array != y] *= t
            weights[predict_array == y] /= t

            # if error == 0:
            #     self.alphas.append(1)
            #     print 'stop!!!!!!!!!!!'
            #     break
            # alpha = 0.5 * np.log((1 - error) / error)
            # print iter_num, error, alpha
            # self.alphas.append(alpha)
            # new_weights = weights * np.exp(-alpha * y * predict_array)
            # # print weights
            # weights = new_weights / np.sum(new_weights)
        self.alphas = np.array(self.alphas)


class DecisionStump(object):
    def __init__(self):
        pass

    def fit(self,X,y,weights=None):
        if weights == None:
            weights = np.ones(X.shape[0])
        min_error = np.inf
        best_feature_index = 0
        best_split_value = 0
        best_sign = -1

        for feature_index in range(X.shape[1]):
            split_values = [np.inf] + np.sort(X[:,feature_index]).tolist()
            # print split_values
            # np.insert(split_values,0,-np.inf)
            # print split_values.shape
            for i in range(len(split_values) - 1):
                split_value = 0.5 * (split_values[i] + split_values[i+1])
                # print split_value

                for sign in [-1,1]:
                    predict_array = np.ones(X.shape[0])
                    if sign == 1:
                        predict_array[X[:,feature_index] < split_value] = -1
                    else:
                        predict_array[X[:, feature_index] > split_value] = -1

                    error = np.sum(weights[predict_array != y])
                    if error < min_error:
                        best_feature_index = feature_index
                        best_split_value = split_value
                        best_sign = sign
                        min_error = error

        self.feture_index = best_feature_index
        self.split_value = best_split_value
        self.sign = best_sign
        return self

    def predict(self,X):
        y = np.ones(X.shape[0])
        if self.sign == 1:
            y[X[:,self.feture_index] < self.split_value] = -1
        else:
            y[X[:, self.feture_index] > self.split_value] = -1
        return y




if __name__ == "__main__":
    x,y = read_data(r'G:\Program\python\machine learning\data\hw2_adaboost_train.dat')
    x_test,y_test = read_data(r'G:\Program\python\machine learning\data\hw2_adaboost_test.dat')
    # print x
    # print y
    model = AdaBoost(max_iter = 1)
    model.fit(x,y)
    print score(x_test,y_test,model.predict)


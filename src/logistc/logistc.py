__author__ = 'Administrator'
# coding=utf-8
import numpy
import math
class logistc:
    def __init__(self):
        self.data_mat = []
        self.label_mat = []
    def load_training_data(self,file_name):
        self.data_mat = []
        self.label_mat = []
        with open(file_name,"r") as f:
            for line in f:
                line_list = line.strip().split()
                line_list = [float(num) for num in line_list]

                self.data_mat.append([1.0] + line_list[0:-1])
                self.label_mat.append(line_list[-1])

    def sigmod(self,x):
        # print x
        return 1.0/(1 + numpy.exp(-x))

    def grad_ascent(self,max_iter = 1000):
        data_matrix = numpy.mat(self.data_mat)
        label_matrix = numpy.mat(self.label_mat).transpose()
        alpha = 0.01
        m,n = numpy.shape(data_matrix)
        weights = numpy.ones((n,1))
        for i in range(max_iter):
            h = self.sigmod(data_matrix*weights)
            error = label_matrix - h
            weights = weights + alpha * data_matrix.transpose()*error
        return weights

logistc = logistc()
logistc.load_training_data("testSet.txt")
# print logistc.data_mat
print logistc.grad_ascent(2000)



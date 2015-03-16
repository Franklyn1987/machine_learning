__author__ = 'Administrator'
# coding=utf-8
import numpy
import math
import random
import matplotlib.pyplot as plt
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
        return 1.0/(1 + numpy.exp(-x))

    # 普通梯度下降
    def grad_ascent(self,max_iter = 1000):
        data_matrix = numpy.mat(self.data_mat)
        label_matrix = numpy.mat(self.label_mat).transpose()
        alpha = 0.01
        m,n = numpy.shape(data_matrix)
        weights = numpy.ones((n,1))
        graph = []
        for i in range(n):
            graph.append([])

        for i in range(max_iter):
            h = self.sigmod(data_matrix*weights)
            error = label_matrix - h
            sigma = alpha * data_matrix.transpose()*error
            # if abs(sigma.sum()) < 1e-6:
            #     print sigma.sum()
            #     print i
            #     break
            weights = weights + sigma
            for index,weight in enumerate(weights):
                graph[index].append(weight[0,0])

        for i in range(len(graph)):
            plt.figure(i)
            # print range(len(graph[i]))
            print len(graph[i])
            plt.plot(range(len(graph[i])),graph[i], 'o')

        plt.show()

        return weights

    # 普通随机梯度下降
    def random_grad_ascent(self,max_iter = 1000):
        data_matrix = numpy.mat(self.data_mat)
        label_matrix = numpy.mat(self.label_mat)
        alpha = 0.01
        # print "label:",numpy.shape(label_matrix)

        m,n = numpy.shape(data_matrix)
        weights = numpy.ones((n,1))



        graph = []
        for i in range(n):
            graph.append([])

        for j in range(max_iter):
            for i in range(m):
                # print weights
                # print data_matrix[i]
                h = self.sigmod(data_matrix[i]*weights)
                # print "h:",h
                error = label_matrix[0,i] - h
                # print "error",numpy.shape(error),error
                # print "data_matrix",numpy.shape(data_matrix[i])
                sigma = alpha * error * data_matrix[i]
                weights = weights + sigma.transpose()
            for index,weight in enumerate(weights):
                graph[index].append(weight[0,0])

        for i in range(len(graph)):
            plt.figure(i)
            # print range(len(graph[i]))
            print len(graph[i])
            plt.plot(range(len(graph[i])),graph[i], 'o')

        plt.show()



        return weights


    def adv_random_grad_ascent(self,max_iter = 1000):
        data_matrix = numpy.mat(self.data_mat)
        label_matrix = numpy.mat(self.label_mat)
        alpha = 0.01
        # print "label:",numpy.shape(label_matrix)

        m,n = numpy.shape(data_matrix)
        weights = numpy.ones((n,1))
        random_list = range(m)

        graph = []
        for i in range(n):
            graph.append([])

        for j in range(max_iter):
            random.shuffle(random_list)
            for i,index in enumerate(random_list):
                # print weights
                # print data_matrix[i]
                # print rand_index
                alpha = 4/(1.0+j+i)+0.01
                # print index
                h = self.sigmod(data_matrix[index]*weights)
                # print "h:",h
                error = label_matrix[0,index] - h
                # print "error",numpy.shape(error),error
                # print "data_matrix",numpy.shape(data_matrix[i])
                sigma = alpha * error * data_matrix[index]
                weights = weights + sigma.transpose()
            for index,weight in enumerate(weights):
                graph[index].append(weight[0,0])

        for i in range(len(graph)):
            plt.figure(i)
            # print range(len(graph[i]))
            print len(graph[i])
            plt.plot(range(len(graph[i])),graph[i], 'o')

        plt.show()

        return weights

logistc = logistc()
print logistc.sigmod(100)
# logistc.load_training_data("testSet.txt")
# # print logistc.data_mat
# print logistc.grad_ascent(2000)
# print logistc.random_grad_ascent(500)
# print logistc.adv_random_grad_ascent(200)


# coding=utf-8
# created by WangZhe on 2014/12/20
import random
import numpy as np
import matplotlib.pyplot as plt

class SVM():
    def __init__(self):
        self.data_mat = []
        self.label_mat = []
        self.alpha = []
    def load_training_data(self, file_name):

        with open(file_name, "r") as f:
            for line in f:
                line_list = line.strip().split()
                line_list = [float(num) for num in line_list]

                self.data_mat.append(line_list[0:-1])
                self.label_mat.append(-1.0 if line_list[-1] == 0 else 1.0)
        return self.data_mat,self.label_mat

    def select_j(self,i,m):
        j = i
        while(j == i):
            j =  random.randint(0,m-1)
        return j

    def clip_alpha(self,a,H,L):
        if a > H:
            a = H
        if L > a:
            a = L
        return a

    def train_kernel(self,X,A,type):
        m,n = np.shape(X)
        K = np.mat(np.zeros((m,1)))
        if type == 'lin':
            temp = (1 + X * A.T)
            K = np.multiply(temp,temp)
            return K


    def smo_simple(self,dataMatIn,classLabels,C,toler,maxIter):
        data_matrix = np.mat(dataMatIn)
        label_matrix = np.mat(classLabels).transpose()
        print data_matrix
        self.data_matrix = data_matrix
        self.label_matrix = label_matrix


        b = 0
        m,n = np.shape(data_matrix)
        print m,n
        K = np.mat(np.zeros((m,m)))

        for i in range(m):
            K[:,i] = self.train_kernel(data_matrix,data_matrix[i,:],'lin')


        alphas = np.mat(np.zeros((m,1)))
        iter = 0
        while iter < maxIter:
            alphaPairsChanged = 0
            print iter
            for i in range(m):

                fXi = float(np.multiply(alphas,label_matrix).T*(data_matrix*data_matrix[i,:].T)) + b
                Ei = fXi - float(label_matrix[i])
                if ((label_matrix[i] * Ei < -toler) and (alphas[i] < C)) or ((label_matrix[i]*Ei > toler) and (alphas[i] > 0)):
                    # print i
                    j = self.select_j(i,m)
                    fXj = float(np.multiply(alphas,label_matrix).T*(data_matrix*data_matrix[j,:].T)) + b
                    Ej = fXj - float(label_matrix[j])
                    alphaIold = alphas[i].copy()
                    alphaJold = alphas[j].copy()
                    if label_matrix[i] != label_matrix[j]:
                        L = max(0,alphas[j]-alphas[i])
                        H = max(C,C + alphas[j]-alphas[i])
                    else:
                        L = max(0,alphas[j] + alphas[i] - C)
                        H = max(C,alphas[j]-alphas[i])
                    if L == H:
                        continue
                    eta = K[i,i] + K[j,j] \
                           - 2.0 * K[i,j]
                    if eta <= 0:
                        continue
                    alphas[j] += label_matrix[j] * (Ei - Ej)/eta
                    alphas[j] = self.clip_alpha(alphas[j],H,L)
                    if abs(alphas[j] - alphaJold) < toler:
                        continue
                    alphas[i] += label_matrix[j] * label_matrix[i] * (alphaJold - alphas[j])
                    b1 = b - Ei - label_matrix[i] * (alphas[i]-alphaIold)*K[i,i]\
                    -label_matrix[j]*(alphas[j]-alphaJold)*K[i,j]
                    b2 = b - Ej - label_matrix[i] * (alphas[i] - alphaIold) * K[i,j] \
                         - label_matrix[j] * (alphas[j] - alphaJold) * K[j,j]
                    if 0 < alphas[i] and C > alphas[i]:
                        b = b1
                    elif 0 < alphas[j] and C > alphas[j]:
                        b = b2
                    else:
                        b = (b1 + b2) * 0.5
                    alphaPairsChanged += 1
            if alphaPairsChanged == 0:
                iter += 1
                # pass
                # print max_iter
                # break
            else:
                iter == 0

        return b,alphas

    def show_svm(self,alphas,b):

        print self.data_matrix.shape
        for i in xrange(self.data_matrix.shape[0]-1):
            if self.label_matrix[i] < 0 :
                plt.plot(self.data_matrix[i,0],self.data_matrix[i,1],'or')
            else:
                plt.plot(self.data_matrix[i,0], self.data_matrix[i,1], 'ob')

        # w = np.zeros((2,1))
        supportVectorsIndex = np.nonzero(alphas.A != 0)[0]
        w = np.zeros((2, 1))
        for i in supportVectorsIndex:
            w += np.multiply(alphas[i]*self.label_matrix[i],self.data_matrix[i,:].T)

        # for i in supportVectorsIndex:
        #     plt.plot(self.data_matrix[i, 0],self.data_matrix[i, 1], 'oy')

        min_x = min(self.data_matrix[:, 0])[0, 0]
        max_x = max(self.data_matrix[:, 0])[0, 0]
        min_y = float(-b -w[0]*min_x)/w[1]
        max_y = float(-b -w[0]*max_x)/w[1]
        plt.plot([min_x,min_y],[max_x,max_y],'-g')

        plt.show()


if __name__ == "__main__":
    model = SVM()
    data,label = model.load_training_data('G:\Program\python\machine learning\src\logistc\\testSet2.txt')

    b,alphas = model.smo_simple(data,label,9999999999999999999999,0.0001,20)
    print alphas
    model.show_svm(alphas,b)
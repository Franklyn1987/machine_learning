# coding=utf-8
# created by WangZhe on 2015/1/11
from sklearn import svm
import numpy as np
from numpy import linalg as LA
import math
import random
def read_data(file,default_label):
    y = []
    x = []
    with open(file,'r') as f:
        for line in f:
            label,feature = line.strip().split("   ")
            label = int(float(label))
            label = 0 if label == default_label  else 1
            features = feature.strip().split("  ")
            features = [float(num) for num in features]
            y.append(label)
            x.append(features)

    x = np.array(x)
    y = np.array(y)
    return x,y

def kernel(x1,x2):
    return math.exp(-100*LA.norm(x1-x2,2)**2)

def get_h(model):
    support_vectors = model.support_vectors_
    alpha = model.dual_coef_[0]

    sum_alpha = sum([abs(x) for x in alpha])
    sum_right = 0
    for i,xi in enumerate(support_vectors):
        for j,xj in enumerate(support_vectors):
            # sum_right += alpha[i] * alpha[j]*kernel(xi,xj)
            sum_right += alpha[i] * alpha[j] * math.exp(-100*LA.norm(xi-xj,2)**2)
    print sum_alpha,sum_right
    return sum_alpha -0.5*sum_right






if __name__ == "__main__":
    x_test,y_test = read_data("G:\\Program\\python\\machine learning\\src\\features.test",0)
    x_train, y_train = read_data("G:\\Program\\python\\machine learning\\src\\features.train", 0)
    train = zip(x_train,y_train)
    import collections
    score = collections.defaultdict(int)

    for i in range(100):
        random.shuffle(train)
        train_set = train[:1000]
        x_train = map(lambda x: x[0], train_set)
        y_train = map(lambda x: x[1], train_set)

        validation_set = train[1000:]
        x_validation = map(lambda x: x[0], validation_set)
        y_validation = map(lambda x: x[1], validation_set)

        # model = svm.SVC(C=0.1, kernel='poly', degree =2).fit(x_validation, y_validation)

        max = [0,0]
        for gramma in [1,10,100,1000,10000]:
            model = svm.SVC(C=0.1,kernel='rbf',gamma=gramma).fit(x_train,y_train)
            result = model.score(x_train,y_train)
            if result > max[1]:
                max = gramma,result
        gramma,max_score = max
        print i,gramma,max_score
        score[gramma] += 1
        


    for key in score:
        print key,score[key]




    #
    # print model.coef_
    # print LA.norm(model.coef_,2)
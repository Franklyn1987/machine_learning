# coding=utf-8
# created by WangZhe on 2015/2/8
from __future__ import division
import numpy as np
import collections
import matplotlib as plt

class ClassifyTree():
    def fit(self,X,y):
        data = np.column_stack((X,y))
        self.index = 0
        self.tree = self.make_tree(data)


    def predict(self,X):
        cur_tree = self.tree

        while(True):
            cur_left = cur_tree[0]
            cur_right = cur_tree[1]
            cur_val = cur_tree[2]
            if cur_left == None and cur_right == None:
                return cur_val
            min_feature_index, min_middle_number = cur_val
            if X[min_feature_index] < min_middle_number:
                cur_tree = cur_left
            else:
                cur_tree = cur_right





    def cal_split_gini(self,data,i,theta):
        data1 = data[data[:,i] < theta]
        data2 = data[data[:,i] >= theta]
        # print 'data1',data1
        # print 'data2',data2
        def cal_gini(data):
            # print 'data:',data
            labels = collections.defaultdict(float)
            for label in data[:,-1]:
                # print label
                labels[label] += 1


            # print i,theta,labels.keys(),labels.values()
            total_num = sum(labels.values())
            gini = 1.0 - sum([ (x /total_num)**2 for x in labels.values()])

            return gini
        gini1 = cal_gini(data1)
        gini2 = cal_gini(data2)
        total_gini = gini1 * data1.shape[0] / data.shape[0] + gini2 * data2.shape[0] / data.shape[0]
        # print data,data.shape
        # print data1,data1.shape
        # print data2,data2.shape
        # print gini1,data1.shape[0]*1.0/data.shape[0],gini2,data2.shape[0]*1.0/data.shape[0]
        # print 'total_gini',total_gini
        return total_gini

    def make_tree(self,data):
        # print self.index
        self.index += 1
        # if self.index == 800:
        #     print 'data',data

        m,n = data.shape

        #判断标签yi是否全部相等，或者xi是否全部相等，相等则为叶子结点，直接返回。
        # print 'data',data
        if (data[:,-1] == data[0][-1]).all() or (data[:,0:-1] == data[0,0:-1]).all():
            return [None,None,data[0][-1]]

        min_feature_index = None
        min_gini = None
        min_middle_number = None

        for feature_index in range(0,n-1):
            sortd_x = data[:,feature_index].tolist()
            sortd_x = sorted(sortd_x)
            if len(sortd_x) == 1:
                continue
            new_mids = [(sortd_x[i] + sortd_x[i + 1]) * 0.5 for i in range(len(sortd_x) - 1)]
            # print mids
            # new_mids = [sortd_x[0] - 100] + mids + [sortd_x[-1] + 100]
            # new_mids = mids
            for middle_number in new_mids:
                gini = self.cal_split_gini(data,feature_index,middle_number)
                if min_gini == None or gini < min_gini:
                    min_gini = gini
                    min_feature_index = feature_index
                    min_middle_number = middle_number
                    # print "index",feature_index,"middle_number",middle_number,"gini",gini

        # print "final_index",feature_index
        # print 'final_middle_number',min_middle_number

        data1 = data[data[:, min_feature_index] < min_middle_number]
        data2 = data[data[:, min_feature_index] >= min_middle_number]
        # print 'data',data
        # print 'data1',data1
        # print 'data2',data2
        left = self.make_tree(data1)
        right = self.make_tree(data2)

        return [left, right, (min_feature_index,min_middle_number)]



if __name__ == "__main__":
    tree = ClassifyTree()
    #
    x_matrix,y_matrix = read_data('hw3_train.dat')
    # x = x[:3]
    # y = y[:3]
    # print x_matrix.shape,y_matrix.shape

    tree.fit(x_matrix,y_matrix)
    # print tree.tree

    data = np.array([[ 0.757222,0.633831,-1.0],[ 0.847382,0.281581,-1.0],[ 0.24931,0.618635,1.0 ]])

    right = 0
    wrong = 0

    x_matrix,y_matrix = read_data('hw3_test.dat')

    for x,y in zip(x_matrix,y_matrix):
        if  tree.predict(x) != y:
            wrong += 1
        else:
            right += 1
    print wrong/(right+wrong)


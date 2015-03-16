__author__ = 'Administrator'
# coding=utf-8
import sys
import math
import types
from collections import defaultdict
import copy
class Maxent:
    def __init__(self):
        self.features = defaultdict(lambda :defaultdict(float))
        self.e_refs = defaultdict(lambda :defaultdict(float))

        self.labels = set()

        self.N = 0
        self.C = 0
        self.train_data = []
        self.tol = 1e-5




    def p_y_x(self,features):
        p_y_x = [(label,math.exp(sum([self.features[feature][label] for feature in features ])) )for label in self.labels]
        z_x = sum([value for label,value in p_y_x])
        p_y_x = [(label,value/z_x) for label,value in p_y_x]
        return p_y_x


    def Eq(self):
        Eq = defaultdict(lambda :defaultdict(float))
        for label,features in self.train_data:
            p_y_x = self.p_y_x(features)
            print p_y_x
            for label,value in p_y_x:
                for feature in features:
                    Eq[feature][label] += value* 1.0 /self.N
        print Eq
        return Eq


        # z_x = sum( [math.exp(value) for item in features for value in self.features[item] ] )
        # p_y_x = math.exp(sum([self.features[item][label] for item in features]))/z_x
        # p_x = sum([value for item in features for values in self.e_refs[item] for value in values])/self.N
        # return p_y_x * p_x


    def load_training_data(self,file_name):
        for line in open(file_name,"r"):
            fields = line.strip().split(" ")
            label = fields[0]
            features = set(fields[1:])
            self.labels.add(label)
            for feature in features:
                self.e_refs[feature][label] += 1
            self.N += 1
            if len(fields[1:]) > self.C:
                self.C = len(fields[1:])
            self.train_data.append((label,features))


        for feature in self.e_refs:
            for label in self.labels:
                self.e_refs[feature][label] = float(self.e_refs[feature][label])/self.N




    def is_end(self,old_features,new_features):
        for old_feature,new_feature in zip(old_features,new_features):
            for label in self.labels:
                if abs(old_features[old_feature][label] - new_features[new_feature][label]) >= self.tol:
                    return False
        return True





    def train(self,max_iter = 1000 ):
        assert len(self.e_refs) > 0
        assert self.N > 0
        assert max_iter > 0
        for i in range(max_iter):
            print i
            self.e_qs = self.Eq()
            old_features = copy.deepcopy(self.features)
            for feature in self.features:
                for label in self.labels:
                    print label,feature,self.e_refs[feature][label],self.e_qs[feature][label]

                    if self.e_refs[feature][label] > 0:
                    # if self.e_refs[feature][label] == 0:
                    #     self.e_refs[feature][label] = 0.0000001


                        delta = (1.0/self.C) * ( math.log(self.e_refs[feature][label])- math.log(self.e_qs[feature][label]))

                    # if self.e_refs[feature][label] >= 0:
                    #     delta = self.newton(self.e_qs[feature][label],self.e_refs[feature][label],self.features[feature][label],self.tol)

                        self.features[feature][label] += delta

            if self.is_end(old_features,self.features):
                break

    def predict(self,features):
        result = self.p_y_x(features)
        sorted(result,key=lambda x:[1],reverse=True)
        print result
        return result

    def newton(self,e_qs,e_ref,theta,tol):
        sigma2 =  2.0
        x0 = 0.0
        x = 0.0
        max_iter = 50
        for i in xrange(100):
            t = e_qs * (math.exp(self.C * x0))
            f = t + (x0 + theta)/sigma2 -e_ref
            g = self.C * t + 1/sigma2
            x = x0 - f/g
            if abs(x - x0) < tol:
                return x
            x0 = x
        raise Exception("Failed to converge after 50 iterations in newton() method")


maxent = Maxent()
maxent.load_training_data("data/train_data.txt")
maxent.train(100)
# for feature in maxent.features:
#     for label in maxent.labels:
#         print feature,label,maxent.features[feature][label]
maxent.predict(["Rainy"])





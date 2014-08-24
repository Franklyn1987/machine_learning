__author__ = 'Administrator'
# coding=utf-8

def f(x):
    return x**2-4

def g(x):
    return 2*x

def cal():
    x = 0.1
    for i in xrange(100):
        dis = - f(x)/g(x)
        if abs(dis) < 1e-20:
            print i
            break
        x += dis
    return x

print cal()

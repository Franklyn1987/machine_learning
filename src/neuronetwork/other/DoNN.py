import numpy as np
from NNLib import neuroNetwork

def readFile(name):
    f = open(name)
    data = []
    for aLine in f:
        elements =aLine.split()
        data.append( [float(elements[0]), float(elements[1]), float(elements[2])])
    data = np.array(data)
    return data

def calEOut(model, data):
    eOut = 0.
    for entry in data:
        input  = entry[0:2]
        answer = entry[2]

        guess = np.sign(model.eval(input))
        if abs(guess-answer)>0.1: eOut +=1
    eOut = eOut / len(data)
    return eOut

def doTraining(model, data, eta):
    for t in range(50000):
        ridx = np.random.randint(0, len(data))
        input  = data[ridx][0:2]
        answer = data[ridx][2]
        model.train(input, answer, eta)

#MAIN
trainData = readFile(r"G:\Program\python\machine learning\data\hw4_nnet_train.dat")
testData  = readFile(r"G:\Program\python\machine learning\data\hw4_nnet_test.dat")

nnStruct = [2,3,1]
r   = 0.1
eta = 0.1

eOutList6 = []
for i in range(500):
    print i
    nn = neuroNetwork( nnStruct, r)
    doTraining(nn, trainData, eta)
    eOutList6.append(calEOut(nn, testData))
print nnStruct, np.average(eOutList6)
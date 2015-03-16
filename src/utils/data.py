# coding=utf-8
# created by WangZhe on 2015/3/1
import numpy as np
def read_data(file_name):
    xs = []
    ys = []
    with open(file_name,'r') as f:
        for line in f:
            line = line.strip().split(' ')
            line = [float(num) for num in line]
            x = line[:-1]

            y = line[-1]
            xs.append(x)
            ys.append(y)

    return np.array(xs),np.array(ys).transpose()

def score(X,y,predict):
    predict_array = predict(X)
    return 1.0 * np.sum(y == predict_array)/X.shape[0]
    # wrong = 0
    # right = 0
    # for x, y_matrix in zip(X, y):
    #     if int(predict(x)) != int(y_matrix):
    #         wrong += 1
    #     else:
    #         right += 1
    # return right *1.0 / (right + wrong)

if __name__ == "__main__":
    pass

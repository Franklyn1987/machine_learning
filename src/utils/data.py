# coding=utf-8
# created by WangZhe on 2015/3/1
import numpy as np
def read_data(file_name,is_y = True):
    xs = []
    ys = []
    with open(file_name,'r') as f:
        for line in f:
            line = line.strip().split(' ')
            line = [float(num) for num in line]
            if is_y:
                x = line[:-1]
                y = line[-1]
                ys.append(y)
            else:
                x = line
            xs.append(x)

    if is_y:
        return np.array(xs),np.array(ys).transpose()
    else:
        return np.array(xs)


def check_arrays(X):
    if X.ndim == 1:
        x_array = np.array([X])
    else:
        x_array = X
    return x_array

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

def score_square(X,y,predict):
    predict_array = predict(X)
    # print predict_array
    compare_array = abs(y - predict_array)
    # print compare_array
    return 1.0 * np.sum(compare_array < 0.1) / X.shape[0]

    # return np.sum(compare_array[compare_array > 0.1]**2)
    # return 1.0 * np.sum(abs(y - predict_array) < 0.1) / X.shape[0]
if __name__ == "__main__":
    pass

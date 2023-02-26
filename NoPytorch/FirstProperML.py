import random as rd
#import numpy as np
#import pandas as pd
#import keras as kr

def h(x, theta):
    s = 0
    for i, j in zip(x, theta):
        s += i * j
    return s

def ReadFileData(File, Dimension):
    Data = list()
    for line in File.readlines():
        v = line.replace('\n', '').split('\t')
        Data.insert(Data.__len__(), ([list(), float(v[Dimension])]))
        Data[Data.__len__() - 1][0].insert(0, 1)
        for i in range(Dimension):
            Data[Data.__len__() - 1][0].insert(i + 1, float(v[i]))
    return Data

def GetAccuracy(Data, theta):
    Accuracy = 0
    for v in Data:
        y = 0
        Accuracy += (v[1] - h(v[0], theta)) ** 2
    return Accuracy

def GradientDescent(Data, alpha, Accuracy):
    Iterations = 20
    theta = list()
    for i in range(Data[0][0].__len__()):
        theta.insert(i, 0)

    while GetAccuracy(Data, theta) > Accuracy:
        for i in range(Iterations):
            DoTrainingExample(rd.choice(Data), theta, alpha)
    
    print("theta's obtained by gradient descent:")
    for i, value in enumerate(theta):
        print("\ttheta{:} = {:.3f}".format(i, value))

def DoTrainingExample(v, theta, alpha):
    delta = h(v[0], theta) - v[1]
    for i in range(theta.__len__()):
        theta[i] -= alpha * delta * v[0][i]

if __name__ == '__main__':
    Data = ReadFileData(open("DataFiles/0.72_0.11_0.72_1.00_0.52_0.07_.dat", "r"), 5)
    GradientDescent(Data, 0.01, 0.000001)
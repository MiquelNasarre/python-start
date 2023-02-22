#import numpy as np
import random as rd
import os

def GenerateVector(Dimension):
    v = list()
    for i in range(Dimension + 1):
        v.insert(v.__len__(), rd.randint(0,100) / 100)
    return v

def RandomDataSet(deviation, InputDimension, Samples):
    try:
        os.mkdir("DataFiles")
    except:
        pass
    theta = GenerateVector(InputDimension)
    filename = ""
    for value in theta:
        filename += "{:.2f}_".format(value)
    File = open("DataFiles/{}.dat".format(filename), "w")
    
    for ignore in range(Samples):
        x = GenerateVector(InputDimension)
        x[0] = 1
        y = 0
        for i , j in zip(x , theta):
            y += i * j
        for i in x[1:]:
            File.write("{:.2f}\t".format(i))
        File.write("{:.4f}\n".format(y + deviation * (rd.randint(0,100) / 100 - 0.5)))
    return theta, "DataFiles/{}.dat".format(filename)

if __name__ == '__main__':
    RandomDataSet(0,5,100)
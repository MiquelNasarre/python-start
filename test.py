import NoPytorch.FirstProperML
import NoPytorch.RandomDataSetGeneration as npt
import Pytorch.pythorchContact as pt1

"""
deviation = 0.01
dimension = 20
sampleSize = 500
alpha = 0.01
accuracy = 0.005

theta , filename = rg.RandomDataSet(deviation, dimension, sampleSize)
print("Generated theta's with deviation {}:".format(deviation))
for i, value in enumerate(theta):
    print("\ttheta{:} = {:.2f}".format(i, value))

ml.GradientDescent(ml.ReadFileData(open(filename, "r"), dimension), alpha, accuracy) 
"""

#pt1.LinearGradDescent_example()

test = list()
test.__add__(1)
#test[0].__add__(1)
print(test[0])

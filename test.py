import FirstProperML as ml
import RandomDataSetGeneration as rg

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

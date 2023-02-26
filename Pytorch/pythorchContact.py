if __name__ == '__main__':
    import LRmodel_class as lr
else:
    import Pytorch.LRmodel_class as lr
import torch as th
import torch.nn as nn
import random as rd

dimensions = 5
Iterations = 1000
tracking = 100
deviation = 0.01
training_step = 0.1

def generate_variables(dim):
    return th.rand(dimensions)

def get_training_example(varbs):
    x = th.rand(dimensions)
    return x, (x * varbs).sum() + (2 * rd.random() - 1) * deviation

def LinearGradDescent_example():
    # Generates the linear function variables we want to find
    varbs = generate_variables(dimensions)

    # Creates the linear regression model we'll use to find them
    Model = lr.model(dimensions, training_step)

    # Loops training
    for epoch in range(Iterations + 1):

        # Generates a training example
        x, y = get_training_example(varbs)

        # Improves the model with the training example
        Model.LearningStep(x, y)

        # Model control printing
        if(epoch % tracking == 0):
            print(f'epoch {epoch}:\tloss = {Model.loss:.8f}')
        
    #Prints results
    print(f'Real variables:\n {varbs}')
    print(f'Trained thetas:\n {Model.theta}')

if __name__ == '__main__':
    LinearGradDescent_example()
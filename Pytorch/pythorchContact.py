if __name__ == '__main__':
    import LRmodel_class as lr
    import SimpleNeuralNetwork_class as snn
else:
    import Pytorch.LRmodel_class as lr
    import Pytorch.SimpleNeuralNetwork_class as snn
import torch as th
import torch.nn as nn
import random as rd

input_dimensions = 5
output_dimensions = 5
Iterations = 20000
tracking = 100
deviation = 0.01
training_step = 0.5
neural_network_layout = [input_dimensions, 10, 10, 10, output_dimensions]

# Example generator functions
def generate_variables(dim):
    return th.rand(dim)

def linear_training_example(varbs):
    x = th.rand(input_dimensions)
    return x, (x * varbs).sum() + 2 * deviation * (rd.random() - 0.5)

def non_linear_training_example():
    x = th.rand(input_dimensions)
    y = 0
    for i in range(input_dimensions):
        y += x[i] ** (i + 1)
    if y > 1:
        return x, th.tensor([1, 0, 0, 0, 1])
    else:
        return x, th.tensor([0, 1, 0, 0, 1])

# Gradient descent functions
def NonLinearGradDescent_example():
    # Creates the linear regression model we'll use to approximate the training set
    Model = snn.model(training_step, neural_network_layout)

    # Loops training
    for epoch in range(Iterations + 1):

        # Generates a training example
        x, y = non_linear_training_example()

        # Improves the model with the training example
        Model.LearningStep(x, y)

        # Model control printing
        if(epoch % tracking == 0):
            print(f'epoch {epoch}:\tloss =\t\t{Model.loss:.8f}')
            print(f'\t\tx =\t\t{x}')
            print(f'\t\ty =\t\t{y}')
            print(f'\t\thypothesis =\t{Model.hypothesis}\n')

    #Prints results
    print(f'Trained thetas:\n {Model.theta}')

    #Model.hypothesis_f(th.tensor([0,0.5,0.5,0.5,0.5], dtype = th.float32))
    #print(Model.hypothesis)

def LinearGradDescent_example():
    # Generates the linear function variables we want to find
    varbs = generate_variables(input_dimensions)

    # Creates the linear regression model we'll use to find them
    Model = lr.model(input_dimensions, training_step)

    # Loops training
    for epoch in range(Iterations + 1):

        # Generates a training example
        x, y = linear_training_example(varbs)

        # Improves the model with the training example
        Model.LearningStep(x, y)

        # Model control printing
        if(epoch % tracking == 0):
            print(f'epoch {epoch}:\tloss = {Model.loss:.8f}')
        
    #Prints results
    print(f'Real variables:\n {varbs}')
    print(f'Trained thetas:\n {Model.theta}')

if __name__ == '__main__':
    NonLinearGradDescent_example()
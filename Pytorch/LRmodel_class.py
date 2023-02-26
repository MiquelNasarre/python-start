import torch as th
import torch.nn as nn

class model:
    def __init__(self, dim, alpha):
        self.theta = th.rand(dim, requires_grad = True)
        self.optimizer = th.optim.SGD([self.theta], lr = alpha)

    def loss_f(self, y):
        self.loss = (self.hypothesis - y) ** 2
    
    def hypothesis_f(self, x):
        self.hypothesis = (x * self.theta).sum()
    
    def grad_loss(self):
        self.loss.backward()

    def LearningStep(self, x, y):
        # Makes the hypothesis for x
        self.hypothesis_f(x)

        # Calculates the loss given the correct answer y
        self.loss_f(y)

        # Calculates the gradient of the thetas
        self.grad_loss()

        # Applies gradient descent
        self.optimizer.step()
        self.optimizer.zero_grad()
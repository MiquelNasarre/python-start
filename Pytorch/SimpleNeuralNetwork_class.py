import torch as th
import torch.nn as nn

class model:
    def __init__(self, alpha, per_layer):
        self.layout = per_layer
        self.theta = list()
        self.optimizer = list()
        for i, thts in enumerate(self.layout[:-1]):
            self.theta.insert(i, list())
            self.optimizer.insert(i, list())
            for j in range(self.layout[i + 1]):
                self.theta[i].insert(j, th.rand(thts, requires_grad = True))
                self.optimizer[i].insert(j, th.optim.SGD([self.theta[i][j]], lr = alpha))

    def g(self, x):
        return 1 / (1 + x.exp())

    def hypothesis_f(self, x):
        a = list()
        a.insert(0, x)
        for i in range(self.layout.__len__() - 1):
            a.insert(i + 1, th.zeros(self.layout[i + 1]))
            for j in range(self.layout[i + 1]):
                a[i + 1][j] = self.g((a[i] * self.theta[i][j]).sum())
        self.hypothesis = a[-1]
        
    def loss_f(self, y):
        self.loss = ((self.hypothesis - y) ** 2).sum()

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
        for i in range(self.layout.__len__() - 1):
            for j in range(self.layout[i + 1]):
                self.optimizer[i][j].step()
                self.optimizer[i][j].zero_grad()

    def FullTrainingSetStep(self, training_set):
        for x, y in training_set:
            self.hypothesis_f(x)
            self.loss_f(y)
            self.grad_loss()
        for i in range(self.layout.__len__() - 1):
            for j in range(self.layout[i + 1]):
                self.optimizer[i][j].step()
                self.optimizer[i][j].zero_grad()
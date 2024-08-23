import torch
import torch.nn.functional as F
import numpy as np
from game_objectives.abstract_objective import AbstractObjective


class SimpleMomentObjective(AbstractObjective):
    def __init__(self):
        AbstractObjective.__init__(self)

    def calc_objective(self, g, f, x, z, y):
        g_of_x = torch.squeeze(g(x))
        f_of_z = torch.squeeze(f(z))
        y = torch.squeeze(y)
        moment = f_of_z.mul(y - g_of_x).mean()
        return moment, -moment


class NormalizedMomentObjective(AbstractObjective):
    def __init__(self, lambda_1=100.0, lambda_2=1.0, lambda_3=0.0):
        AbstractObjective.__init__(self)
        self._lambda_1 = lambda_1
        self._lambda_2 = lambda_2
        self._lambda_3 = lambda_3

    def __str__(self):
        return ("NormalizedObjective::lambda_1=%f:lambda_2=%f:lambda_3=%f"
                % (self._lambda_1, self._lambda_2, self._lambda_3))

    def calc_objective(self, g, f, x, z, y):
        epsilon = y - torch.squeeze(g(x))
        f_of_z = torch.squeeze(f(z))
        # f_of_z = f_of_z - f_of_z.mean()
        raw_moment = f_of_z.mul(epsilon).mean()
        denominator = (f_of_z ** 2).mul(epsilon ** 2).mean() ** 0.5
        # denominator = denominator.detach()
        moment_norm = raw_moment / denominator
        # print(denominator)
        f_reg = self._lambda_1 * (F.relu(f_of_z.abs() - 0.3)).mean()
        f_reg += self._lambda_2 * (f_of_z.mean() ** 2)
        # f_reg /= denominator
        # g_reg = 0.02 * ((epsilon - epsilon.mean()) ** 2).mean()
        g_reg = self._lambda_3 * F.relu((epsilon ** 2).mean() ** 0.5 - 1.0)
        # f_reg = 0
        # g_reg = 0

        return moment_norm + g_reg, -moment_norm + f_reg


class RegularizedMomentObjective(AbstractObjective):
    def __init__(self, lambda_1=0.1, lambda_2=1.0, lambda_3=0.0):
        AbstractObjective.__init__(self)
        self._lambda_1 = lambda_1
        self._lambda_2 = lambda_2
        self._lambda_3 = lambda_3

    def __str__(self):
        return ("RegObjective::lambda_1=%f:lambda_2=%f:lambda_3=%f"
                % (self._lambda_1, self._lambda_2, self._lambda_3))

    def calc_objective(self, g, f, x, z, y):
        g_of_x = torch.squeeze(g(x))
        f_of_z = torch.squeeze(f(z))
        y = torch.squeeze(y)
        moment = f_of_z.mul(y - g_of_x).mean()
        regularizer_1 = torch.nn.functional.mse_loss(f_of_z - f_of_z.mean(), torch.tensor(5.0).double().to(moment.device)) 
        regularizer_2 = f_of_z.mean()**2
        return moment, -moment + self._lambda_1*regularizer_1 + self._lambda_2*regularizer_2


class HingeRegularizedMomentObjective(RegularizedMomentObjective):
    def __str__(self):
        return ("HingeRegObjective::lambda_1=%f:lambda_2=%f:lambda_3=%f"
                % (self._lambda_1, self._lambda_2, self._lambda_3))

    def calc_objective(self, g, f, x, z, y):
        g_of_x = torch.squeeze(g(x))
        f_of_z = torch.squeeze(f(z))
        y = torch.squeeze(y)
        moment = f_of_z.mul(y - g_of_x).mean()
        regularizer_1 = (torch.nn.functional.relu(f_of_z.abs()-0.3)).mean() 
        regularizer_2 = f_of_z.mean()**2
        # g_reg = 100.0 * (F.relu(y - g_of_x).abs() - 0.3).mean()
        g_reg = self._lambda_3 * F.relu(((y - g_of_x) ** 2).mean() ** 0.5 - 1.0)
        return moment + g_reg, -moment + self._lambda_1*regularizer_1 + self._lambda_2*regularizer_2


class OptimalMomentObjective(AbstractObjective):
    def __init__(self, lambda_1=0.25):
        AbstractObjective.__init__(self)
        self._lambda_1 = lambda_1

    def __str__(self):
        return "OptimalObjective::lambda_1=%f" % self._lambda_1
    
    def calc_objective(self, g, f, x, z, y,g_global,f_global,w_locals_prev):
        g_of_x = torch.squeeze(g(x))
        f_of_z = torch.squeeze(f(z))
        y = torch.squeeze(y)
        epsilon = g_of_x - y
        ans_g=0
        ans_f=0
        moment = f_of_z.mul(epsilon).mean()
        f_reg = self._lambda_1 * (f_of_z ** 2).mul(epsilon ** 2).mean()
        g_reg = 0.0
        if g_global!=0 and w_locals_prev:
          for i in g_global:
              ans_g += np.linalg.norm(g_global[i]-w_locals_prev[0][i])**2
        #   ans_g = np.linalg.norm(g_global['cnn.0.weight']-w_locals_prev[0]['cnn.0.weight']) ** 2 + np.linalg.norm(g_global['cnn.0.bias']-w_locals_prev[0]['cnn.0.bias']) ** 2 + np.linalg.norm(g_global['model.2.weight']-w_locals_prev[0]['model.2.weight']) ** 2 + np.linalg.norm(g_global['model.2.bias']-w_locals_prev[0]['model.2.bias']) ** 2 + np.linalg.norm(g_global['model.4.weight']-w_locals_prev[0]['model.4.weight']) ** 2 + np.linalg.norm(g_global['model.4.bias']-w_locals_prev[0]['model.4.bias']) ** 2
          ans_f = np.linalg.norm(f_global['model.0.weight']-w_locals_prev[1]['model.0.weight']) ** 2 + np.linalg.norm(f_global['model.0.bias']-w_locals_prev[1]['model.0.bias']) ** 2 + np.linalg.norm(f_global['model.2.weight']-w_locals_prev[1]['model.2.weight']) ** 2 + np.linalg.norm(f_global['model.2.bias']-w_locals_prev[1]['model.2.bias']) ** 2 
        return moment + g_reg + (0.001/2)*(ans_g) , -moment + f_reg + (0.001/2)*ans_f
        # return (epsilon ** 2).mean(), -moment + f_reg

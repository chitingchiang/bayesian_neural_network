import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
from torch.optim import SGD

class VariationalLinear(nn.Module):
    def __init__(self, in_features, out_features, init_sigma=1):
        super(VariationalLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.init_sigma = init_sigma

        self.W_mu = Parameter(torch.randn(self.out_features, self.in_features)*self.init_sigma)
        self.b_mu = Parameter(torch.randn(self.out_features)*self.init_sigma)

        self.W_rho = Parameter(torch.ones(self.out_features, self.in_features)*1e-4)
        self.b_rho = Parameter(torch.ones(self.out_features)*1e-4)

    def forward(self, X):
        #W_sigma = F.softplus(self.W_rho)
        #b_sigma = F.softplus(self.b_rho)
        #W_sigma = torch.exp(self.W_rho)
        #b_sigma = torch.exp(self.b_rho)
        W_sigma = torch.abs(self.W_rho)
        b_sigma = torch.abs(self.b_rho)

        W = self.W_mu+W_sigma*torch.randn(self.out_features, self.in_features)
        b = self.b_mu+b_sigma*torch.randn(self.out_features)

        return F.linear(X, W, b), (self.W_mu, W_sigma, W, self.b_mu, b_sigma, b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, init_sigma={}'.format(
            self.in_features, self.out_features, self.init_sigma
        )

class VariationalDense(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=[], init_sigma=1):
        super(VariationalDense, self).__init__()

        features = [in_features]+hidden_features+[out_features]

        model = []
        for i in range(len(features)-1):
            model.append(VariationalLinear(in_features=features[i],
                                           out_features=features[i+1],
                                           init_sigma=init_sigma))
        self.model = nn.ModuleList(model)

    def forward(self, X):
        params_list = []
        for layer in self.model[:-1]:
            X, params = layer(X)
            X = F.relu(X)
            params_list.append(params)
        X, params = self.model[-1](X)
        params_list.append(params)
        return X, params_list

if __name__=='__main__':
    in_neuron = 1
    out_neuron = 1
    hidden_neurons = [4, 8, 4]

    model = VariationalDense(in_features=in_neuron,
                             out_features=out_neuron,
                             hidden_features=hidden_neurons,
                             init_sigma=0.5)
    print(model)

    n_batch = 32
    X = torch.randn(n_batch, 1)
    yhat, params_list = model(X)

    print(yhat.size())
    print(len(params_list))

    for params in params_list:
        print(params[0].size(), params[1].size(), params[2].size(),
              params[3].size(), params[4].size(), params[5].size())

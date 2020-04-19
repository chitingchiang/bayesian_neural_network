# Bayesian neural network

In this repository, we explore the Bayesian neural network,
which is a form of variational inference. In the standard
neural network, we only estimate the values of the model
parameters that minimize the loss function. However, knowing
the uncertainty of the model is also crucial for many applications,
and the Bayesian neural network allows us to simultaneously
obtain the bet parameters and their uncertainties.

In our implementation, we assume that all parameters follow
Gaussian distributions, so that each of them can be parametrized
by the mean and standard deviation. Since the standard deviation
has to be positive, we try three different parametrizations:
(1) <img src="https://latex.codecogs.com/gif.latex?\sigma_p={\rm softplus}(\sigma_\rho) " /> 
(2) <img src="https://latex.codecogs.com/gif.latex?\sigma_p=\exp(\sigma_\rho) " />
(3) <img src="https://latex.codecogs.com/gif.latex?\sigma_p=|\sigma_\rho| " />


In [the first notebook](linear_regression.ipynb), we train the
Bayesian neural network with a single neuron with linear activation.
The network contains four parameters: mean and standard deviation
of the weight and bias. We compare the results with the standard
linear regression. Specifically, we find a good agreement between
the standard deviation of the parameters fitted from the Bayesian
neural network and the ones obtained from the linear regression
with bootstrapped sample. Interestingly, training the Bayesian
neural network (variational inference: approximating the posterior
to an analytic distribution and minimizing the KL divergence between
the approximated posterior and the product of the prior and likelihood
distributions) requires the knowledge of data variance (for computing
the likelihood), but this is not required for bootstrapping and
linear regressing the sample.

In [the second notebook](nonlinear_regression.ipynb)



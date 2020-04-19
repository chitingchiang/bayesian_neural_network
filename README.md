# Bayesian neural network

In this repository, we explore the Bayesian neural network,
which is a form of variational inference. In the standard
neural network, we only estimate the values of the model
parameters that minimize the loss function. However, knowing
the uncertainty of the model is crucial for many applications,
and the Bayesian neural network allows us to simultaneously
obtain the bet parameters and their uncertainties.

In our implementation, we assume that all parameters follow
Gaussian distributions, so that each of them can be parametrized
by the mean (mu_p) and standard deviation (sigma_p). The Bayesian
neural network thus has twice as much parameters as for the
standard neural network. For the forward propagation, we first
sample each parameter using the Gaussian distribution with mu_p
and sigma_p, and then follow the standard neural netowrk computation
using the sampled weights and biases. Even with the same mu_p and
sigma_p, every time we will sample a different parameter by chance,
and as a result the prediction will be different for every forward
propagation. This allows us to compute the uncertainty of the
prediction. Note that the standard neural network can be understood
as the Bayesian neural network with sigma_p=0.

To train the Bayesian neural network, we approximate the posterior
(parameter distribution given data) to certain analytic distribution
(product of Gaussian in our implementation) and minimize the KL
divergence between the approximated posterior and the product
of the prior and likelihood distributions. In our implementation,
we adopt a Gaussian likelihood, and for the prior we explore both
flat (no preferred values for any parameters) and Gaussian with
zero mean. The Gaussian prior with zero mean is equivalent to the
ridge regression.

Since the standard deviation of the parameter has to be positive,
we try three different parametrizations: (1) sigma_p = softplus(sigma_rho);
(2) sigma_p = exp(sigma_rho); (3) sigma_p = |sigma_rho|. With the
above parametrizations, there is no constraint on sigma_rho. We
obtain the best convergence using the third parametrization with
the initial value set to be 0.0001. The intuition for initializing
sigma_rho with small values is that the Bayesian neural network can
first optimize for mu_p.

In [the first notebook](linear_regression.ipynb), we train the
Bayesian neural network with a single neuron with linear activation
on a sample generated from a linear relation. The Bayesian neural
network contains four parameters: mean and standard deviation of
the weight and bias. We compare the results with the standard
linear regression and find a good agreement between the standard
deviation of the parameters fitted from the Bayesian neural network
and the ones obtained from the linear regression with bootstrapped
sample. The agreement also holds for the ridge regression when Lambda
is non-zero. Interestingly, training the Bayesian neural network
requires the knowledge of data variance (for computing the likelihood),
but this is not required for bootstrapping and linear regressing
the sample.

In [the second notebook](nonlinear_regression.ipynb), we train the
neural network with multiple layers on a nonlinear sample. We assume
the data to be y = sin(x) + epsilon where epsilon is the Gaussian
noise. We first train the standard neural network with two four-neuron
layers and find that the model can fit the data reasonably well.
A better fit can be obtained with more neurons and more layers,
but the overfitting becomes more severe and the convergence is
much more difficult to achieve. We then train the Bayesian neural
network using the same architecture, but note that now the amount
of parameters is doubled. With the trained Bayesian neural network,
we are able to estimate the uncertainty of the prediction, and
we find that the uncertainty is larger at the region where there
is no sample. This agrees with our intuition and demonstrates the
power of the Bayesian neural network.

In conclusion, the biggest advantage of the Bayesian neural network
is the ability to quantify the prediction uncertainty. The price we
pay is the amount of parameters (doubled). From our experience, the
biggest issue is the convergence. It generally take many more training
steps for the Bayesian neural network than the standard neural network
to achieve the same quality of the final fit. Also, the parametrization
of sigma_p can have a large impact on the result.


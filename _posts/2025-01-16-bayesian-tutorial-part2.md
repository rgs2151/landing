---
layout: default
title: "Bayesian Modeling - Part 2: Bayesian Thinking and Model Specification"
date: 2025-01-16
tag: thoughts
---

Now that we understand basic probability and continuous distributions, let's dive into the heart of Bayesian modeling. We'll learn how to think like a Bayesian and build our first complete model.

## Probability Interpretation: Two Schools of Thought

Choosing the right distribution is a question of what fits the data best. **When we are approximating the data, we are fitting a model!**

There are 2 schools of thought on this:

1. **Frequentist**: They consider the data as fixed (given the experiment) and view the model parameters as unknown but fixed quantities. They use methods like Maximum Likelihood Estimation (MLE) to estimate the parameters that maximize the likelihood of observing the data.

2. **Bayesian**: They consider the data as observed, and the **model parameters as random variables with prior beliefs**. They use **Bayes' theorem to update their prior beliefs** about the parameters into posterior distributions based on the observed data.

**For the purpose of this tutorial, we will focus specifically on the Bayesian approach.**

## Bayesian Model Specification

**Bayesian model specification involves 3 components:**

1. **Prior**: The prior distribution describes the beliefs about the model parameters before observing the data. It is often assumed to be a normal distribution.
2. **Likelihood**: The likelihood function describes the probability of observing the data given the model parameters. It is the foundation of the model.
3. **Posterior**: The posterior distribution describes the updated beliefs about the model parameters after observing the data. It is calculated using Bayes' theorem.

We can update our beliefs about the generative process by using the posterior distribution:

$$Posterior = \frac{Likelihood \times Prior}{Evidence}$$

### Our First Bayesian Model

**For our example of the balls in the bag, we can specify the model as follows:**

**Prior**:
 
$$\mu \sim \mathcal{N}(0, 1)$$ 
$$\sigma \sim |\mathcal{N}(0, 1)|$$

**Likelihood**:

$$y \sim \mathcal{N}( \mu, \sigma^2)$$

**Posterior**:

$$P(\mu, \sigma | y) = \frac{P(y | \mu, \sigma) P(\mu) P(\sigma)}{P(y)}$$

**Where**:
- $y$ is the data (balls drawn from the bag).
- $\mu$ is the mean of the distribution.
- $\sigma$ is the standard deviation of the distribution.
- $\mathcal{N}$ is the normal distribution.

A nice way to visualize this is through a simple graphical model:

*[Image Caption: A directed graphical model showing μ and σ as parent nodes pointing to y, with a plate notation indicating N observations y₁, y₂, ..., yₙ]*

This indicates that there are $ N $ independent and identically distributed (i.i.d.) observations $ y_1, y_2, \dots, y_N $.

## Inference: Solving the Bayesian Model

Yayy, now that we have everything, *LETS GO AND SOLVE IT!!!!*

But wait, how do we solve it?

Solving means calculating the posterior distribution $P(\mu, \sigma | y)$.

- **Analytically**: We can solve the posterior distribution analytically for simple models such as this.
- **Numerically**: For complex models, it is often **intractable to solve** the posterior distribution analytically. In such cases, we can use numerical methods like **Markov Chain Monte Carlo (MCMC)** or **Variational Inference (VI)** to approximate the posterior distribution.

**For the purpose of this tutorial, we will use MCMC to approximate the posterior distribution.**

### Implementing Our First Model with Stan

Let's implement and solve our first Bayesian model using Stan:

```python
import numpy as np
import tempfile
from cmdstanpy import CmdStanModel
import arviz as az
import matplotlib.pyplot as plt

# Define the Stan model
model_specification = """
data {
    int<lower=0> N; // Number of data points
    array[N] real y; // Data
}
parameters {
    real mu; // Mean
    real<lower=0> sigma; // Standard deviation
}
model {
    // Priors
    mu ~ normal(0, 1);
    sigma ~ normal(0, 1);

    // Likelihood
    y ~ normal(mu, sigma);
}
"""

# Write the model to a temporary file
with tempfile.NamedTemporaryFile(suffix=".stan", mode="w", delete=False) as tmp_file:
    tmp_file.write(model_specification)
    tmp_stan_path = tmp_file.name

# Prepare the unknown generator data
data = {
    "N": 1000,
    "y": np.random.normal(loc=1, scale=1, size=1000),
}

# Compile and fit the model
model = CmdStanModel(stan_file=tmp_stan_path)
fit = model.sample(data=data, iter_sampling=1000, step_size=0.1)
```

Now let's examine our results:

```python
idata = az.from_cmdstanpy(fit)

# Plot the posterior distribution
ax = az.plot_posterior(idata, round_to=2, figsize=(8, 2), textsize=10)
plt.suptitle("Posterior Distribution of Parameters")
for a in ax: 
    a.set_ylabel("Density")
plt.show()
```

**Notes**:
- It seems like the model has updated its prior beliefs and landed on ~1.1 for $\mu$ and ~0.9 for $\sigma$ which is very close to the true distribution of the generative process (1 $\mu$ and 1 $\sigma$).
- The model is uncertain about its estimation only to a very small degree. The range for $\mu$ is 1 to 1.1 and for $\sigma$ is 0.95 to 1. It is essentially pretty certain about these parameters.

Let's visualize how well our model fits the data:

```python
def get_color_gradient(n, color1="red", color2="blue"):
    cmap = mcolors.LinearSegmentedColormap.from_list("gradient", [color1, color2])
    return [cmap(i / (n - 1)) for i in range(n)]

num_bins = 26
all_mu = idata.posterior.mu.values.flatten()
all_sigma = idata.posterior.sigma.values.flatten()

x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(6, 3))

# Plot the data histogram
_, _, patches = plt.hist(data["y"], bins=num_bins, density=True, 
                        edgecolor='black', color="skyblue", alpha=0.7, label="Data")
for i in range(len(patches)): 
    patches[i].set_facecolor(get_color_gradient(num_bins)[i])

# Plot the uncertainty in the model
for i in range(len(all_mu)):
    pdf = norm.pdf(x, loc=all_mu[i], scale=all_sigma[i])
    plt.plot(x, pdf, color="orange", alpha=0.003)

# Plot the mean of the posterior predictive distribution
mean_pdf = norm.pdf(x, loc=all_mu.mean(), scale=all_sigma.mean())
plt.plot(x, mean_pdf, "k--", label="Mean Post. Predictive")

plt.legend(loc='upper left')
plt.title("Posterior Predictive Distribution and Uncertainty")
plt.ylabel("Density")
plt.xlabel("y")
plt.xlim(-4, 4)
plt.show()
```

## Understanding the Results

The orange cloud shows the uncertainty in our model predictions - each thin orange line represents one possible parameter combination from our posterior samples. The thick dashed black line shows the average prediction.

This uncertainty quantification is one of the key advantages of Bayesian modeling. We don't just get point estimates; we get full distributions that tell us how confident we should be in our predictions.

## Key Takeaways

1. **Bayesian models have three components**: Prior, Likelihood, and Posterior
2. **Priors encode our beliefs** before seeing data
3. **Likelihoods describe** how our data depends on parameters
4. **Posteriors combine** priors and likelihoods using Bayes' theorem
5. **MCMC helps us solve** complex models numerically
6. **Uncertainty quantification** comes naturally in Bayesian analysis

## What's Next?

In Part 3, we'll explore **Mixture Models** - what happens when our data comes from multiple subpopulations? We'll learn how to model situations where different groups follow different distributions, all mixed together in our observed data.

Think about it: what if our bag of balls actually contained multiple smaller bags, each with different color preferences? That's exactly what we'll tackle next!

---

*Continue to [Part 3: Mixture Models and Hidden Structure](../bayesian-tutorial-part3) to learn about modeling multiple subpopulations in your data.*

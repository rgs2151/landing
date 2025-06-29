---
layout: default
title: "Bayesian Modeling - Part 5: Input-Dependent Models and GLMs"
date: 2025-01-19
tag: thoughts
---

We've journeyed from simple stationary models to complex temporal dependencies. Now we enter the realm of **Dynamic Models** - where external factors influence our generative process. This is where Bayesian modeling becomes incredibly powerful for real-world applications.

## Linear Regressions and the Idea of Uncertainty

Traditional linear models are based on the equation:
$$
y = x \cdot w + c
$$

This form represents a **deterministic relationship** between the input $x$ and the output $y$. Every input $x$ maps to a single, exact $y$ without any randomness or uncertainty.

In real-world data, however, outputs are often affected by factors not captured by the model (e.g., measurement errors, hidden variables, or inherent variability). Adding noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ acknowledges this uncertainty and makes the model more realistic, leading to:

$$
y = x \cdot w + c + \epsilon
$$

where:
- $x$ is the input (features),
- $w$ is the weight vector,
- $c$ is the intercept (bias),
- $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is the Gaussian noise with mean $0$ and variance $\sigma^2$.

This can be equivalently **reparameterized** as:

$$
y \sim \mathcal{N}(x \cdot w + c, \sigma^2)
$$

This directly states that $y$ is drawn from a normal distribution with mean $x \cdot w + c$ and variance $\sigma^2$ (a typical Bayesian representation).

The noisy version better reflects the variability seen in observed data.

## Example: Temperature-Dependent Ball Colors

What does it mean to be driven by some external factors?

To make the **color of the ball input-dependent**, we can think of **x** as a **temperature scale** (1-dimensional), ranging from a cold "blue" temperature to a warm "red" temperature. The **color of the ball (y)** then depends on this temperature. For instance, at lower temperatures, the balls are more likely to be blue, and as the temperature increases, the likelihood shifts toward red.

Let's generate some data to illustrate this:

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm

# Generate temperature-dependent ball colors
w = 0.5
c = 5
sigma = 2

x = np.random.uniform(-10, 10, size=1000)
y = np.random.normal(loc=x * w + c, scale=sigma)

# Visualize the data
col_line = mcolors.LinearSegmentedColormap.from_list("gradient", ["blue", "red"])
plt.figure(figsize=(6, 3))
plt.scatter(x, y, alpha=0.7, c=y, cmap=col_line, marker="+")
plt.colorbar(ticks=[])
plt.ylabel("y (color of ball)")
plt.xlabel("x (temperature)")
plt.yticks([])
plt.title("Color over temperature")
plt.show()
```

## Model Specification for Linear Regression (GLM)

Let $ x $ represent the temperature, a continuous variable ranging from -10 (coldest) to 10 (hottest).

Modeling the color $y$ of the ball as a linear function of the temperature, we can specify our model as follows:

**Priors**: 
$$w \sim \mathcal{N}(0, 1)$$ 
$$c \sim \mathcal{N}(0, 1)$$
$$\sigma \sim |\mathcal{N}(0, 1)|$$

Notice how we don't put any prior on $x$ and only on the parameters - this is the **core of Bayesian modeling**. The prior distributions are initially chosen to be normal with mean $0$ and variance $1$ for simplicity.

**Likelihood**: 
$$y \sim \mathcal{N}(x \cdot w + c, \sigma^2)$$

**Posterior**:
We will update those priors based on Bayes' theorem:

$$
P(w, c, \sigma | y, x) = \frac{P(y | x, w, c, \sigma) P(w) P(c) P(\sigma)}{P(y)}
$$

Where:
- $P(y | x, w, c, \sigma)$ is the likelihood of observing the data given the parameters.
- $P(w)$, $P(c)$, and $P(\sigma)$ are the prior distributions of the parameters.
- $P(y)$ is the total probability of observing the data.

## Inference in Linear Regression with Stan

Let's solve this with MCMC using Stan:

```python
import tempfile
from cmdstanpy import CmdStanModel
import arviz as az

# Define the Stan model
model_specification = """
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}
parameters {
    real w;
    real c;
    real<lower=0> sigma;
}
model {
    // Priors
    w ~ normal(0, 1);
    c ~ normal(0, 1);
    sigma ~ normal(0, 1);
    
    // Likelihood
    y ~ normal( w * x + c, sigma);
}
"""

# Write the model to a temporary file
with tempfile.NamedTemporaryFile(suffix=".stan", mode="w", delete=False) as tmp_file:
    tmp_file.write(model_specification)
    tmp_stan_path = tmp_file.name

# Prepare the data
data = {
    "N": 1000,
    "x": x,
    "y": y,
}

# Compile and fit the model
model = CmdStanModel(stan_file=tmp_stan_path)
fit = model.sample(data=data, iter_sampling=1000, step_size=0.1)

# Analyze results
idata = az.from_cmdstanpy(fit)
az.plot_posterior(idata, round_to=2, figsize=(6, 2), textsize=10)
plt.suptitle("Posterior Distribution of GLM Parameters")
plt.show()
```

Let's compare our recovered parameters with the true ones:

```python
import pandas as pd
import seaborn as sns

def get_color_gradient(n, color1="red", color2="blue"):
    cmap = mcolors.LinearSegmentedColormap.from_list("gradient", [color1, color2])
    return [cmap(i / (n - 1)) for i in range(n)]

# Compare original vs recovered parameters
plt.figure(figsize=(6, 3))

data = pd.DataFrame({
    "Parameter": ["w", "c", "sigma"] * 2,
    "Type": ["Original"] * 3 + ["Recovered"] * 3,
    "Value": [
        w, c, sigma, 
        idata.posterior.w.values.flatten().mean(), 
        idata.posterior.c.values.flatten().mean(), 
        idata.posterior.sigma.values.flatten().mean()
    ]
})

sns.barplot(
    data=data, 
    x="Parameter", 
    y="Value", 
    hue="Type", 
    palette=get_color_gradient(2),
    dodge=True,
    alpha=0.7,
    edgecolor="black"
)
plt.title("Original vs Recovered GLM Parameters")
plt.show()
```

Now let's visualize the fit with uncertainty:

```python
# Visualize the fit with uncertainty
col_line = mcolors.LinearSegmentedColormap.from_list("gradient", ["blue", "red"])
plt.figure(figsize=(6, 3))
plt.scatter(x, y, alpha=0.7, c=y, cmap=col_line, marker="+")

p_x = np.linspace(-10, 10, 1000)

# Plot the uncertainty in the fit
for i in range(1000):
    w_sample = idata.posterior.w.values[0][i]
    c_sample = idata.posterior.c.values[0][i]
    u_y = w_sample * p_x + c_sample
    plt.plot(p_x, u_y, color="orange", alpha=0.01)

# Plot the mean posterior predictive
p_y = idata.posterior.w.values.flatten().mean() * p_x + idata.posterior.c.values.flatten().mean()
plt.plot(p_x, p_y, color="black", linestyle="--", 
         label="Mean Post. Predictive", linewidth=2)

plt.colorbar(ticks=[])
plt.ylabel("y (color of ball)")
plt.xlabel("x (temperature)")
plt.yticks([])
plt.title("GLM Fit and Uncertainty in Fit")
plt.legend()
plt.show()
```

The orange cloud shows the uncertainty in our linear relationship - this is one of the key advantages of Bayesian GLMs. We don't just get a single line; we get a full distribution of possible relationships.

## Input-Dependent Mixtures: Multiple Factories

***IMPORTANT: PLEASE READ THE EXAMPLE***

Imagine there are **two factories**, each producing balls with specific weight-color patterns:

1. **Factory 1**: Produces balls where heavier balls tend to be redder. The color of the ball increases linearly with its weight but with some variability.
2. **Factory 2**: Produces balls where heavier balls tend to be bluer. The relationship between weight and color is different and noisier than Factory 1.

However, when you receive a shipment of balls, you don't know which factory produced each ball (this is the **hidden state**). All you see are the weights ($ x $) and colors ($ y $) of the balls.

*[Image Caption: Two factory diagrams side by side - Factory 1 showing heavier balls getting redder, Factory 2 showing heavier balls getting bluer, with arrows indicating the linear relationships]*

### Steps explaining the synthetic data generation:

- **Step 1**: Randomly assign balls to factories.
  - Factory 1 (70% of the time).
  - Factory 2 (30% of the time).

- **Step 2**: For each ball, generate a weight ($ x $) uniformly between 1 and 10:
  $$
  x \sim \text{Uniform}(1, 10)
  $$

- **Step 3**: Use the corresponding factory's GLM to generate the ball's color ($ y $):
  - If the ball comes from Factory 1: $ y \sim 0.8 \cdot x + 1.5 + \epsilon $
  - If the ball comes from Factory 2: $ y \sim -0.5 \cdot x - 2.0 + \epsilon $
  - Where $ \epsilon $ is small Gaussian noise to introduce variability.

Let's generate this data:

```python
# Factory parameters
w1, w2 = 0.8, -0.5
c1, c2 = 1, -1
sigma1, sigma2 = 1, 2

# Generate data from each factory
x_1 = np.random.uniform(1, 10, size=300)
x_2 = np.random.uniform(1, 10, size=700)
y_1 = np.random.normal(loc=w1 * x_1 + c1, scale=sigma1)
y_2 = np.random.normal(loc=w2 * x_2 + c2, scale=sigma2)

# Visualize the individual factories
fig, ax = plt.subplots(1, 2, figsize=(10, 3))

cmap1 = mcolors.LinearSegmentedColormap.from_list("gradient", ["blue", "red"])
sc = ax[0].scatter(x_1, y_1, c=y_1, cmap=cmap1, alpha=0.7, marker="+")
fig.colorbar(sc, ticks=[], ax=ax[0])
ax[0].set_ylabel("y (color of ball)")
ax[0].set_yticks([])
ax[0].set_xlabel("x (weight)")
ax[0].set_title("From Factory 1")

cmap2 = mcolors.LinearSegmentedColormap.from_list("gradient", ["red", "blue"])
sc = ax[1].scatter(x_2, 1-y_2, c=1-y_2, cmap=cmap2, alpha=0.7, marker="+")
fig.colorbar(sc, ticks=[], ax=ax[1])
ax[1].set_ylabel("y (color of ball)")
ax[1].set_yticks([])
ax[1].set_xlabel("x (weight)")
ax[1].set_title("From Factory 2")

plt.suptitle("Truth, Hidden. For understanding only. (notice the colorbars)")
plt.show()

# What you actually observe
cmap = mcolors.LinearSegmentedColormap.from_list("gradient", ["blue", "red"])
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
tot_x = np.concatenate([x_1, x_2])
tot_y = np.concatenate([y_1, y_2])
sc = plt.scatter(tot_x, tot_y, c=tot_y, cmap=cmap, alpha=0.7, marker="+")
plt.colorbar()
plt.ylabel("y (color of ball)")
plt.xlabel("x (weight)")
plt.title("What you can Observe")
plt.show()
```

## Mixture of GLM Specification

**Priors**:
- **Mixing Weights**:
  $$
  \pi \sim \text{Dirichlet}(\alpha_1, \alpha_2)
  $$
- **GLM Parameters for Each Component**:
  $$
  w_k \sim \mathcal{N}(0, 1), \quad c_k \sim \mathcal{N}(0, 1), \quad \sigma_k \sim |\mathcal{N}(0, 1)| \quad \text{for } k = 1, 2
  $$

**Likelihood**:
For each observed $ y_n $:
$$
P(y_n | x_n, z_n = k, w_k, c_k, \sigma_k) = \mathcal{N}(x_n \cdot w_k + c_k, \sigma_k^2)
$$

The overall likelihood is a mixture:
$$
P(y_n | x_n, \pi, \{w_k, c_k, \sigma_k\}_{k=1}^2) = \sum_{k=1}^2 \pi_k \mathcal{N}(y_n | x_n \cdot w_k + c_k, \sigma_k^2)
$$

**Posterior**:
Using Bayes' theorem, the posterior is:
$$
P(\pi, \{w_k, c_k, \sigma_k\}_{k=1}^2 | y, x) \propto P(y | x, \pi, \{w_k, c_k, \sigma_k\}_{k=1}^2) P(\pi) \prod_{k=1}^2 P(w_k) P(c_k) P(\sigma_k)
$$

Let's implement this mixture of GLMs in Stan:

```python
# Define the Stan model for mixture of GLMs
model_specification = """
data {
    int<lower=0> N; // Number of data points
    int<lower=0> K; // Number of components

    vector[N] x;
    vector[N] y;
}
parameters {
    simplex[K] pi; // Mixture weights
    vector[K] w; // Slopes
    vector[K] c; // Intercepts
    vector<lower=0>[K] sigma; // Standard deviations
}
model {
    // Priors
    pi ~ dirichlet(rep_vector(1, K));
    w ~ normal(0, 1);
    c ~ normal(0, 1);
    sigma ~ normal(0, 1);
    
    // Likelihood
    for (n in 1:N) {
        vector[K] log_likelihoods;
        for (k in 1:K) {
            log_likelihoods[k] = log(pi[k]) + normal_lpdf(y[n] | x[n] * w[k] + c[k], sigma[k]);
        }
        target += log_sum_exp(log_likelihoods);
    }
}
"""

# Fit the model
with tempfile.NamedTemporaryFile(suffix=".stan", mode="w", delete=False) as tmp_file:
    tmp_file.write(model_specification)
    tmp_stan_path = tmp_file.name

data = {
    "N": 1000,
    "K": 2,
    "x": tot_x,
    "y": tot_y,
}

model = CmdStanModel(stan_file=tmp_stan_path)
fit = model.sample(data=data, iter_sampling=1000, step_size=0.1)

idata = az.from_cmdstanpy(fit)
ax = az.plot_trace(idata, figsize=(8, 4), compact=True)
plt.show()
```

**Notice the label switching!!** All chains look stable so we will stick with one of them to avoid the switching problem.

Let's visualize the recovered GLMs:

```python
# Extract posterior samples (using first chain to avoid label switching)
p_pi = idata.posterior.data_vars["pi"].values[0]
p_w = idata.posterior.data_vars["w"].values[0]
p_c = idata.posterior.data_vars["c"].values[0]
p_sigma = idata.posterior.data_vars["sigma"].values[0]

# Plot the recovered GLMs onto the data
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
cmap = mcolors.LinearSegmentedColormap.from_list("gradient", ["blue", "red"])
sc = plt.scatter(tot_x, tot_y, c=tot_y, cmap=cmap, alpha=0.7, marker="+")

# Plot the uncertainty
x_line = np.linspace(1, 10, 100)
for g in range(2):
    for i in range(0, 1000, 50):  # Sample every 50th to reduce clutter
        pi = p_pi[i, g]
        w = p_w[i, g]
        c = p_c[i, g]
        u_y = w * x_line + c
        plt.plot(x_line, u_y, color="orange", alpha=0.02)

# Plot the mean posterior predictive
p_y_1 = p_w.mean(0)[0] * x_line + p_c.mean(0)[0]
p_y_2 = p_w.mean(0)[1] * x_line + p_c.mean(0)[1]
plt.plot(x_line, p_y_1, color="black", linestyle="--", 
         label="Component 1 Mean", linewidth=2)
plt.plot(x_line, p_y_2, color="black", linestyle="-", 
         label="Component 2 Mean", linewidth=2)

plt.colorbar(sc)
plt.ylabel("y (color of ball)")
plt.xlabel("x (weight)")
plt.title("Mixture of GLMs: Recovered Factory Relationships")
plt.legend()
plt.show()
```

**Quiz**: What would happen if we had 3 factories instead?

## Key Insights About Input-Dependent Models

1. **External Influences**: GLMs allow external factors to influence our distributions
2. **Regression meets Bayesian**: We get uncertainty quantification in our regression parameters
3. **Mixture Extensions**: We can have different relationships for different subgroups
4. **Real-world Relevance**: Most real data has input dependencies!

## What's Next?

In our final part, Part 6, we'll combine everything we've learned into **Input-Dependent Markov Models (GLM-HMMs)**. These are the most sophisticated models in our series, combining temporal dependencies with input-driven dynamics.

Think of conveyor belts in a factory where the belt choice depends on the previous belt, and each belt produces balls with weight-dependent colors. This is where Bayesian modeling gets truly exciting for complex real-world systems!

---

*Continue to [Part 6: GLM-HMMs and Advanced Sequential Modeling](../bayesian-tutorial-part6) to learn about the most sophisticated models combining all concepts we've covered.*

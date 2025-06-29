---
layout: default
title: "Bayesian Modeling - Part 6: GLM-HMMs and Advanced Sequential Modeling"
date: 2025-01-20
tag: thoughts
---

Welcome to the final part of our Bayesian modeling journey! We've traveled from simple balls in bags to complex mixture models and temporal dependencies. Now we'll combine everything into the most sophisticated model in our series: **Input-Dependent Markov Models**, also known as **GLM-HMMs** or **Switching Linear Regression**.

This is where Bayesian modeling truly shines for complex, real-world sequential data with changing dynamics.

## The Conveyor Belt Factory

### Example: Conveyor Belts in the Color Factory

Imagine a **factory** with **three conveyor belts**, each handling different ball-packing operations:

1. **Belt 1 (Red Balls)**:
   - On this belt, the redness of a ball ($ y $) depends positively on its weight ($ x $), so heavier balls are redder.
2. **Belt 2 (Blue Balls)**:
   - Here, the redness depends negatively on the ball's weight, making heavier balls bluer.
3. **Belt 3 (Neutral Balls)**:
   - This belt doesn't correlate much with weight, producing balls of all colors.

### The Markovian Twist

Unlike the mixture model (where balls came independently from factories), this process is sequential:

- The conveyor belts operate in a **Markovian sequence**:
  - If the current belt is Belt 1, it's most likely to switch to Belt 2.
  - If the current belt is Belt 2, it tends to switch to Belt 3.
  - If the current belt is Belt 3, it's likely to switch back to Belt 1.
- You observe the weight ($ x $) and color ($ y $) of each ball in sequence but cannot directly see the belt ($ z $) it came from.

*[Image Caption: Three conveyor belts arranged in a factory setting, with arrows showing the probabilistic transitions between belts. Each belt shows different weight-color relationships - Belt 1 with positive correlation, Belt 2 with negative correlation, Belt 3 with weak correlation]*

## Details on the Synthetic Data Generation

1. **Hidden States ($ z_n $)**:
   - The conveyor belt in use at time $ n $ (hidden variable).
   - Follows a **Markov process**, where:
     $$
     P(z_n | z_{n-1}) = \text{Transition Probability Matrix}
     $$
   - Example transition matrix:
     $$
     \begin{bmatrix}
     0.5 & 0.4 & 0.1 \\ 
     0.1 & 0.5 & 0.4 \\ 
     0.4 & 0.1 & 0.5
     \end{bmatrix}
     $$

2. **Observed Data ($ x, y $)**:
   - $ x $: Weight of the ball ($ x \sim \text{Uniform}(1, 10) $).
   - $ y $: Color of the ball, determined by the GLM specific to the belt ($ z_n $).

3. **GLMs for Emission Components**:
   - Each conveyor belt uses a GLM to model the relationship between weight ($ x $) and color ($ y $):
     - **Belt 1 (Red)**: $ y_n \sim \mathcal{N}(w_1 \cdot x_n + c_1, \sigma_1^2) $
     - **Belt 2 (Blue)**: $ y_n \sim \mathcal{N}(w_2 \cdot x_n + c_2, \sigma_2^2) $
     - **Belt 3 (Neutral)**: $ y_n \sim \mathcal{N}(w_3 \cdot x_n + c_3, \sigma_3^2) $
   - Their values:
     - Belt 1: $ w_1 = 1.0, c_1 = 2.0, \sigma_1 = 1.0 $
     - Belt 2: $ w_2 = -0.8, c_2 = -1.0, \sigma_2 = 1.5 $
     - Belt 3: $ w_3 = 0.2, c_3 = 0.0, \sigma_3 = 0.8 $

4. **Sequence Generation**:
   - Start with an initial belt ($ z_1 \sim \pi $, where $ \pi $ is the initial state distribution).
   - For each time step $ n $:
     1. Transition to the next belt ($ z_n $) based on the current belt ($ z_{n-1} $).
     2. Generate the weight ($ x_n $) from a uniform distribution.
     3. Generate the color ($ y_n $) using the GLM of the current belt ($ z_n $).

Let's start by generating and visualizing our data:

```python
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dynamax.hidden_markov_model import LinearRegressionHMM
from dynamax.utils.utils import find_permutation

# Generator parameters
weights = np.array([1, -0.8, 0.2])
bias = np.array([2, 0, -1])
sigma = np.array([1, 0.8, 1.5])

trans_mat = np.array([
    [0.5, 0.4, 0.1],
    [0.1, 0.5, 0.4],
    [0.4, 0.1, 0.5]
])

initial_probs = np.array([1, 1, 1]) / 3

# First, let's see what happens if we ignore the sequential nature
x = np.linspace(1, 10, 1000)
y_1 = np.random.normal(loc=weights[0] * x + bias[0], scale=sigma[0])
y_2 = np.random.normal(loc=weights[1] * x + bias[1], scale=sigma[1])
y_3 = np.random.normal(loc=weights[2] * x + bias[2], scale=sigma[2])

col_line = mcolors.LinearSegmentedColormap.from_list("gradient", ["blue", "red"])
plt.figure(figsize=(6, 3))
plt.scatter(np.concatenate([x, x, x]), np.concatenate([y_1, y_2, y_3]), 
           alpha=0.7, c=np.concatenate([y_1, y_2, y_3]), cmap=col_line, marker="+")
plt.colorbar(ticks=[])
plt.ylabel("y (color of ball)")
plt.xlabel("x (weight)")
plt.yticks([])
plt.title("If we did not care for the sequence of ball arrival!")
plt.show()
```

As you can see, if the balls all came in the same bag at once, there would not be any Markovian process. In that case it just boils down to a mixture! Now let's make these balls come in sequences:

```python
# Set up the sequential data generation
num_states = 3
emission_dim = 1
covariate_dim = 1
num_timesteps = 100
batch_size = 10

# Initialize the HMM
hmm = LinearRegressionHMM(num_states, covariate_dim, emission_dim)

true_params, _ = hmm.initialize(
    jr.PRNGKey(0),
    initial_probs=jnp.array(initial_probs),
    transition_matrix=jnp.array(trans_mat),
    emission_weights=jnp.array(weights.reshape(-1, 1)),
    emission_biases=jnp.array(bias.reshape(-1, 1)),
    emission_covariances=jnp.array(sigma.reshape(-1, 1, 1))
)

# Create time-varying inputs (sinusoidal weight pattern)
inputs = (jnp.sin(2 * jnp.pi * jnp.arange(num_timesteps) / 50) * 4 + 5).reshape(-1, 1)

# Sample from the true model
true_states, emissions = hmm.sample(true_params, jr.PRNGKey(1), num_timesteps, inputs=inputs)

# Generate multiple batches for training
batch_inputs = jnp.array(np.random.uniform(0, 10, size=(batch_size, num_timesteps, covariate_dim)))
batch_true_states, batch_emissions = [], []

for i in range(batch_size):
    a, b = hmm.sample(true_params, jr.PRNGKey(i+10), num_timesteps, inputs=batch_inputs[i])
    batch_true_states.append(a)
    batch_emissions.append(b)
batch_true_states = jnp.array(batch_true_states)
batch_emissions = jnp.array(batch_emissions)
```

Let's visualize the sequential data:

```python
plt.figure(figsize=(12, 3))
col_line = mcolors.LinearSegmentedColormap.from_list("gradient", ["blue", "red"])
plt.scatter(jnp.arange(num_timesteps), emissions, c=emissions, cmap=col_line, marker='o', s=100)
plt.colorbar()
plt.ylabel("Emission Color")
plt.xlabel("Time")
plt.title("If you were given this, could you tell which belt each point came from?")
plt.show()

# Plot the inputs, emissions, and true states
fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 4]}, figsize=(12, 4))

em_max = emissions.max() + 2
em_min = emissions.min() - 2

# Plot inputs (weights)
axs[0].plot(inputs, 'k-', marker='.')
axs[0].set_xlim(0, num_timesteps)
axs[0].set_ylim(0, 10)
axs[0].set_ylabel("weight")

# Plot emissions with true states as background
axs[1].imshow(true_states[None, :],
              extent=(0, num_timesteps, em_min, em_max),
              aspect="auto",
              cmap="Grays",
              alpha=0.5)
axs[1].plot(emissions, 'k--', marker='.')
axs[1].scatter(jnp.arange(num_timesteps), emissions, c=emissions, cmap=col_line, marker='o', s=100)
axs[1].set_xlim(0, num_timesteps)
axs[1].set_ylim(em_min, em_max)
axs[1].set_ylabel("emissions (color)")
axs[1].set_xlabel("Time")

plt.suptitle("True Simulated Data")
plt.show()
```

**Can you tell the belts apart now? 🔎**

## GLM-HMM Specification (Switching Linear Regression)

**1. Emissions (GLM for Observed Data)**:
- The observed data $ y_t $ at time $ t $ is modeled as a Gaussian distribution:
  $$
  y_t \mid x_t, z_t = k \sim \mathcal{N}(x_t \cdot w_k + c_k, \sigma_k^2)
  $$
  where:
  - $ z_t \in \{1, 2, \dots, K\} $: Discrete latent state (which conveyor belt is active at time $ t $).
  - $ x_t $: Input feature (ball weight).
  - $ w_k $: Weight for the linear relationship in state $ k $.
  - $ c_k $: Bias (intercept) for state $ k $.
  - $ \sigma_k^2 $: Variance of the Gaussian noise for state $ k $.

**2. Latent States (Markov Process)**:
- The discrete latent states $ z_t $ follow a **Markov process**:
  $$
  z_1 \sim \pi, \quad z_{t+1} \mid z_t \sim P_{z_t}
  $$
  where:
  - $ \pi $: Initial state distribution.
  - $ P_{z_t} $: Transition probability matrix describing transitions between states.

**Priors on Model Parameters**:
$$
\pi \sim \text{Dirichlet}(\alpha_{\pi})
$$
$$
P_{z_t} \sim \text{Dirichlet}(\alpha_P)
$$
$$
w_k \sim \mathcal{N}(0, 1), \quad c_k \sim \mathcal{N}(0, 1) \quad \text{for } k = 1, \dots, K
$$
$$
\sigma_k \sim |\mathcal{N}(0, 1)| \quad \text{for } k = 1, \dots, K
$$

**Likelihood**:

For each observed $ y_t $:
1. Conditioned on the hidden state $ z_t $, the likelihood is:
   $$
   P(y_t \mid x_t, z_t = k, w_k, c_k, \sigma_k) = \mathcal{N}(y_t \mid x_t \cdot w_k + c_k, \sigma_k^2)
   $$
2. The full likelihood marginalizes over all possible hidden states:
   $$
   P(y_t \mid x_t, \pi, P, \{w_k, c_k, \sigma_k\}_{k=1}^K) = \sum_{k=1}^K P(z_t = k \mid z_{t-1}, P) \cdot \mathcal{N}(y_t \mid x_t \cdot w_k + c_k, \sigma_k^2)
   $$

**Posterior**:

Using Bayes' theorem, the posterior is:
$$
P(\pi, P, \{w_k, c_k, \sigma_k\}_{k=1}^K \mid y_{1:T}, x_{1:T}) \propto P(y_{1:T} \mid x_{1:T}, \pi, P, \{w_k, c_k, \sigma_k\}_{k=1}^K) \cdot P(\pi) \cdot P(P) \cdot \prod_{k=1}^K P(w_k) P(c_k) P(\sigma_k)
$$

## How to Solve This Complex Model? 😱

We can solve it the same way we have been solving our models. However, the **Markovian nature** of the model introduces **dependencies** between the observations, making it more complex to solve than the previous models. Making full Bayesian inference computationally **VERY** expensive.

For practical purposes, we'll use the **Expectation-Maximization (EM)** algorithm with optimizations:

1. **Parameter Estimation**: Estimate parameters by maximizing the marginal likelihood
2. **Forward-Backward Algorithm**: Efficiently compute state probabilities 
3. **Viterbi Algorithm**: Find the most likely sequence of states

**Let's go and train it!**

```python
# Initialize and fit the GLM-HMM
test_params, param_props = hmm.initialize(jr.PRNGKey(42))

# Fit the model using EM
test_params, lps = hmm.fit_em(test_params, param_props, batch_emissions, inputs=batch_inputs)

# Plot training progress
plt.figure(figsize=(6, 3))
plt.plot(lps, "k--", label='EM Training')
plt.xlabel('Epoch')
plt.ylabel('Log Probability')
plt.title("GLM-HMM Training Progress")
plt.legend()
plt.show()
```

Now let's see how well our model recovered the true states:

```python
def plot_states(true_states, most_likely_states, emissions, title="Did the HMM get it right?"):
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 2]}, figsize=(12, 6))

    for i, states in enumerate([true_states, most_likely_states]):
        axs[i].imshow(states[None, :],
                    extent=(0, num_timesteps, em_min, em_max),
                    aspect="auto",
                    cmap="binary",
                    alpha=0.5)
        axs[i].plot(emissions, 'k-', marker='.')
        axs[i].scatter(jnp.arange(num_timesteps), emissions, c=emissions, 
                      cmap=col_line, marker='o', s=100)
        axs[i].set_xlim(0, num_timesteps)
        axs[i].set_ylim(em_min, em_max)
        axs[i].set_ylabel("emissions")
        axs[i].set_xlabel("time")

    axs[0].set_title("true states")
    axs[1].set_title("inferred states")
    plt.suptitle(title)

# Compute the most likely states
most_likely_states = hmm.most_likely_states(test_params, emissions, inputs=inputs)
plot_states(true_states, most_likely_states, emissions)
plt.show()
```

It looks pretty bad, doesn't it? But look closer - this is our old friend **LABEL SWITCHING**! Let's align them:

```python
permutation = find_permutation(true_states, most_likely_states)
remapped_states = permutation[true_states]
plot_states(remapped_states, most_likely_states, emissions, title="How about now after alignment?")
plt.show()
```

Much better! 

**QUIZ**: Why does it get certain time points exceptionally wrong? (Even after label switching, why is it extra difficult?) Hint: Look at the input closely.

## Parameter Recovery Analysis

Let's examine how well we recovered the original parameters:

```python
import seaborn as sns
import pandas as pd

def get_color_gradient(n, color1="red", color2="blue"):
    cmap = mcolors.LinearSegmentedColormap.from_list("gradient", [color1, color2])
    return [cmap(i / (n - 1)) for i in range(n)]

# Extract recovered parameters
t_tr = np.array(test_params.transitions.transition_matrix)
t_init = np.array(test_params.initial.probs).reshape(-1)
t_weights = np.array(test_params.emissions.weights).reshape(-1)
t_covs = np.array(test_params.emissions.covs).reshape(-1)
t_bias = np.array(test_params.emissions.biases).reshape(-1)

# Compare transition matrices
fig, ax = plt.subplots(1, 2, figsize=(8, 3))

# Apply permutation to true parameters for comparison
true_trans_aligned = trans_mat[permutation, :][:, permutation]

sns.heatmap(true_trans_aligned, ax=ax[0], cmap="Greens", annot=True, cbar=False, vmax=1, vmin=0)
ax[0].set_title("True Transition Matrix")

sns.heatmap(t_tr, ax=ax[1], cmap="Greens", annot=True, cbar=False, vmax=1, vmin=0)
ax[1].set_title("Recovered Transition Matrix")
plt.show()

# Compare GLM parameters
def plot_param_comparison(param, r_param, title, ylabel, ax=None, width=0.35):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    states = np.arange(1, len(param) + 1)
    x = np.arange(len(param))
    colors = get_color_gradient(2)
    
    ax.bar(x - width / 2, param, width, label='Generator', color=colors[0], alpha=0.7)
    ax.bar(x + width / 2, r_param, width, label='Recovered', color=colors[1], alpha=0.7)
    ax.set_xticks(x, [f'State {i}' for i in states])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('States')
    ax.legend()

fig, ax = plt.subplots(2, 2, figsize=(10, 6))

# Plot each parameter comparison with permutation applied
plot_param_comparison(weights[permutation], t_weights, "Weights Comparison", "Weight Value", ax=ax[0, 0])
plot_param_comparison(bias[permutation], t_bias, "Bias Comparison", "Bias Value", ax=ax[0, 1])
plot_param_comparison(sigma[permutation], t_covs, "Sigma Comparison", "Sigma Value", ax=ax[1, 0])
plot_param_comparison(initial_probs[permutation], t_init, "Initial Probabilities Comparison", "Probability", ax=ax[1, 1])

plt.tight_layout()
plt.show()
```

**Considering these are point estimates from only 10 batches, the results are pretty good!**

## Final Thoughts

Bayesian modeling is both an art and a science. The art lies in choosing appropriate priors and model structures that capture the essence of your problem. The science lies in the rigorous mathematical framework that ensures coherent uncertainty quantification and principled inference.

**Thank you for attending my Ted Talk! 🙇🎤**
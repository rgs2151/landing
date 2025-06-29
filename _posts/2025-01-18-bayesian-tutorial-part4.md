---
layout: default
title: "Bayesian Modeling - Part 4: Markov Models and Temporal Dependencies"
date: 2025-01-18
tag: thoughts
---

Imagine our generative process has evolved again. Instead of sampling directly from a single bag or even from multiple nested bags (as in the mixture model), now we have a **sequence** of bags. The bag we sample from at any given step **depends on which bag we chose in the previous step**.

This situation introduces the concept of **dependencies in time or sequence**, where the current state depends on the previous one. Such a process can be described using **Markov Models**.

## What is a Markov Process?

A **Markov Process** models systems that evolve over time, where:
1. The system can be in one of several **states**.
2. The probability of transitioning to a new state depends only on the **current state** and not on the sequence of states before it. This is the **Markov Property**:
   $$
   P(z_n | z_{n-1}, z_{n-2}, \dots) = P(z_n | z_{n-1})
   $$

In our example, the system moves from one bag to another over time. The bag you pick your ball from at step $ n $ depends only on the bag you chose at step $ n-1 $.

## Introducing Hidden Markov Models (HMMs)

*[Image Caption: A sequence diagram showing three bags connected by arrows, with each bag producing colored balls. The bags represent hidden states, and the balls represent observations]*

A **Hidden Markov Model (HMM)** extends a Markov Process by introducing **observations**. You no longer see the actual bag (state), but only the **ball** drawn from it (observation).

Key components of an HMM:
1. **Hidden States ($ z_n $)**:
   - Represent which "hidden bag" is active at each time step $ n $.
   - Transition between states is governed by the **transition probabilities**.

2. **Observations ($ y_n $)**:
   - Drawn from a Gaussian distribution parameterized by the **mean** ($ \mu_k $) and **variance** ($ \sigma_k^2 $) of the active state:
   $$
   y_n \sim \mathcal{N}(\mu_{z_n}, \sigma_{z_n}^2)
   $$

3. **Transition Probabilities ($ P(z_n | z_{n-1}) $)**:
   - Govern how likely it is to move from one state to another.
   - Captures the sequential dependencies in the generative process.

4. **Emission Probabilities ($ P(y_n | z_n) $)**:
   - Describe the distribution of balls in each bag.
   - E.g., a "red-dominant" bag emits more red balls, but it can still emit blue ones occasionally.

## Generative Process for an HMM

The generative process of an HMM can be described as follows:

1. **Transition**:
   - At time $ n $, transition to a new state $ z_n $ based on the current state $ z_{n-1} $:
   $$
   z_n \sim \text{Categorical}(\pi_{z_{n-1}})
   $$

2. **Emission**:
   - Once in state $ z_n $, generate an observation $ y_n $ from the Gaussian distribution of that state:
   $$
   y_n \sim \mathcal{N}(\mu_{z_n}, \sigma_{z_n}^2)
   $$

### Joint and Marginal Distributions

1. **Joint Probability of Observations and States**:
   $$
   P(y, z) = P(z_1) \prod_{n=2}^N P(z_n | z_{n-1}) P(y_n | z_n)
   $$

2. **Marginal Probability of Observations**:
   - By summing (marginalizing) over all possible hidden states:
   $$
   P(y) = \sum_{z_1, z_2, \dots, z_N} P(y, z)
   $$

You can also imagine this as a dynamic mixture model:

*[Image Caption: A comparison diagram showing static mixture model vs dynamic mixture model (HMM), highlighting how the mixture weights change over time in HMMs]*

## Probabilistic Graphical Model

Below is the PGM illustrating the HMM. It shows the hierarchical dependencies:

*[Image Caption: A directed graphical model showing the temporal chain z₁ → z₂ → z₃ → ... → zₙ, with each zᵢ pointing down to its corresponding observation yᵢ, and μₖ parameters influencing the emissions]*

- $ z_n $: Hidden state at time $ n $, governing the emission and the next state (the bag)
- $ \mu_k$: Parameters of the Gaussian distribution for each hidden state.
- $ y_n $: Observed ball drawn at time $ n $.
- The arrows show how $ z_{n-1} $ influences $ z_n $, and $ z_n $ determines $ y_n $.

## Expectation-Maximization: The Frequentist Approach

The **Expectation-Maximization (EM)** algorithm is a frequentist approach to solving Hidden Markov Models. Unlike full Bayesian inference, EM provides **point estimates** of the parameters by maximizing the likelihood of the observed data.

Key differences from Bayesian approach:
- We won't have estimates for the **posterior distribution** of the parameters or uncertainty (credible intervals).
- EM gives a single "best guess" for the parameters (maximum likelihood estimates).
- EM usually does not use priors on the parameters (but we can add them if we want!).
- EM is computationally faster and simpler since it avoids sampling (e.g., MCMC).

### How EM Solves HMMs

Using **EM**, solving an HMM involves:
1. **E-Step**: Infer the hidden states based on the current parameter estimates.
2. **M-Step**: Update the parameters to maximize the likelihood given the inferred states.
3. **Repeat** until the parameters converge.

## Example: Solving Gaussian HMM with EM

Let's implement and solve an HMM using the Dynamax library. First, let's define our true parameters:

```python
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dynamax.hidden_markov_model import GaussianHMM
import seaborn as sns

# True parameters of the generative process
true_num_states = 3
emission_dim = 1

# Specify parameters of the HMM
initial_probs = np.array([1, 1, 1]) / 3

transition_matrix = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]
])

emission_means = np.array([[-3.0], [0.0], [4.0]])
emission_covs = np.array([[[0.7]], [[0.7]], [[0.7]]])
```

Let's visualize the distributions of each state in the HMM:

```python
def get_color_gradient(n, color1="red", color2="blue"):
    cmap = mcolors.LinearSegmentedColormap.from_list("gradient", [color1, color2])
    return [cmap(i / (n - 1)) for i in range(n)]

def plot_mixture(y, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    
    _, _, patches = ax.hist(y, bins=26, density=True, edgecolor='black', 
                           color="skyblue", alpha=0.7, label="Data")
    for i, patch in enumerate(patches):
        patch.set_facecolor(get_color_gradient(26)[i])
    
    ax.set_title("State Distribution")
    ax.set_xlabel("y")
    ax.set_ylabel("Density")

# Visualize each state's distribution
figs, axs = plt.subplots(1, 3, figsize=(15, 3))

for k in range(true_num_states):
    components = np.random.choice(3, size=1000, p=transition_matrix[k])
    y = np.array([np.random.normal(emission_means[k,0], emission_covs[k,0]) 
                  for k in components])
    plot_mixture(y=y, ax=axs[k])
    axs[k].set_title(f"State {k+1} Distribution")

plt.tight_layout()
plt.show()
```

**Notice how each state has a certain preference for the color of the balls.** These can be considered as state definitions! Notice how each state has got a certain preference for the color of the balls.

Now let's generate some data from this synthetic process and try to solve it:

```python
num_train_batches = 3
num_test_batches = 1
num_timesteps = 200

# Initialize the HMM
g_hmm = GaussianHMM(true_num_states, emission_dim)

# Convert to JAX arrays
transition_matrix = jnp.array(transition_matrix)
emission_means = jnp.array(emission_means)
emission_covs = jnp.array(emission_covs)

true_params, _ = g_hmm.initialize(
    initial_probs=initial_probs,
    transition_matrix=transition_matrix,
    emission_means=emission_means,
    emission_covariances=emission_covs
)

# Sample train and test data
train_key, test_key = jr.split(jr.PRNGKey(0), 2)
f = vmap(partial(g_hmm.sample, true_params, num_timesteps=num_timesteps))
train_true_states, train_emissions = f(jr.split(train_key, num_train_batches))
test_true_states, test_emissions = f(jr.split(test_key, num_test_batches))
```

Let's visualize the generated data:

```python
fig, ax = plt.subplots(3, 1, figsize=(10, 5))
col_line = mcolors.LinearSegmentedColormap.from_list("gradient", ["red", "blue"])

for idx, emission in enumerate(train_emissions):
    ax[idx].plot(emission, c="black", alpha=0.2)
    ax[idx].scatter(np.arange(0, 200, 1), emission, c=emission, 
                   cmap=col_line, marker="+", s=30)
    ax[idx].set_title(f"Chain {idx + 1}")
    ax[idx].set_xlabel("Time")
    ax[idx].set_ylabel("Emission")

plt.tight_layout()
plt.show()
```

## Training the Gaussian HMM using EM

```python
# Initialize the parameters using K-Means
key = jr.PRNGKey(0)
t_hmm = GaussianHMM(num_states=3, emission_dim=1, transition_matrix_stickiness=10.)
params, props = t_hmm.initialize(key=key, method="kmeans", emissions=train_emissions)
params, lps = t_hmm.fit_em(params, props, train_emissions, num_iters=100)

# Plot training progress
plt.figure(figsize=(6, 3))
true_lp = vmap(partial(g_hmm.marginal_log_prob, params))(train_emissions).sum()
true_lp += g_hmm.log_prior(params)
plt.axhline(true_lp, color='k', linestyle=':', 
           label="True LP = {:.2f}".format(true_lp))
plt.plot(lps, "k--", label='EM')
plt.xlabel('num epochs')
plt.ylabel('log prob')
plt.legend()
plt.title("Training Log Probability")
plt.show()
```

**Quiz**: Why does the EM fit to a better log prob than the Generator? Hint: Think about the number of samples.

Let's compare the recovered parameters with the true ones:

```python
from dynamax.utils.utils import find_permutation

# Extract recovered parameters
r_init = np.array(params.initial.probs)
r_tr = np.array(params.transitions.transition_matrix)
r_means = np.array(params.emissions.means)
r_covs = np.array(params.emissions.covs)

# Compare initial probabilities
import pandas as pd

data = pd.DataFrame({
    "State": ["State 1", "State 2", "State 3"] * 2,
    "Type": ["Original"] * 3 + ["Recovered"] * 3,
    "Probability": np.concatenate([initial_probs, r_init])
})

plt.figure(figsize=(6, 3))
sns.barplot(
    data=data, 
    x="State", 
    y="Probability", 
    hue="Type", 
    palette=get_color_gradient(2),
    dodge=True,
    alpha=0.7,
    edgecolor="black"
)
plt.title("Comparison of Original and Recovered Initial Probabilities")
plt.ylabel("Probability")
plt.xlabel("State")
plt.ylim(0, 1)
plt.legend(title="Initial Probabilities")
plt.show()
```

**Quiz**: The recovered init probs are not exactly the same as the generator probs! Can you think of a reason why? What is going on between state 2 and state 3 init probs? Hint: Take a closer look at our training chains!

Let's examine the transition matrices:

```python
# Compare transition matrices
rows = ["Current State 1", "Current State 2", "Current State 3"]
columns = ["Next State 1", "Next State 2", "Next State 3"]

g_tr = np.array(transition_matrix)

fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
row_shade = get_color_gradient(6)

for i, row in enumerate(rows):
    ax = axes[i]
    data = []
    for j, col in enumerate(columns):
        data.append({"Column": col, "Type": "Original", "Value": g_tr[i, j]})
        data.append({"Column": col, "Type": "Recovered", "Value": r_tr[i, j]})
    
    plot_data = pd.DataFrame(data)
    sns.barplot(
        data=plot_data,
        x="Column",
        y="Value",
        hue="Type",
        palette=row_shade[i*2:i*2+2],
        ax=ax,
        edgecolor="black",
        alpha=0.7
    )
    ax.set_title(f"Transition Probs - {row}")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend(title="Transition Probs.")

axes[-1].set_xlabel("Column")
plt.tight_layout()
plt.show()
```

Finally, let's compare the emission distributions:

```python
from scipy.stats import norm
import matplotlib.lines as mlines

x = np.linspace(-5, 8, 1000)
emm_colors = get_color_gradient(3)

plt.figure(figsize=(6, 3))

# Plot original emission distributions
for i in range(3):
    pdf = norm.pdf(x, loc=emission_means[i,0], scale=emission_covs[i,0])
    plt.plot(x, pdf, label=f"Original State {i + 1}", 
            color=emm_colors[i], alpha=0.5)

# Plot recovered emission distributions
for i in range(3):
    pdf = norm.pdf(x, loc=r_means[i,0], scale=r_covs[i,0])
    plt.plot(x, pdf, label=f"Recovered State {i + 1}", 
            color=emm_colors[i], linestyle="--")

plt.title("Emission Distributions")
plt.ylabel("Density")
plt.xlabel("Emission")

# Create custom legend handles
solid_line = mlines.Line2D([], [], color='black', linestyle='-', 
                          alpha=0.5, label='Original')
dashed_line = mlines.Line2D([], [], color='black', linestyle='--', 
                           label='Recovered')

plt.legend(handles=[solid_line, dashed_line], loc="upper left")
plt.show()
```

**Notice: the states are permuted!**

**Quiz**: Can you map the recovered states to generator states (only with critical thinking)?

## Key Insights About Hidden Markov Models

1. **Temporal Dependencies**: HMMs capture how the current state depends on the previous state
2. **Hidden Structure**: We observe the outcomes but not the underlying states
3. **Label Switching**: Parameter recovery can suffer from label switching problems
4. **Sequential Modeling**: Perfect for time series data with changing regimes

## What's Next?

In Part 5, we'll move into **Dynamic Models** and explore **Input-Dependent Distributions (GLMs)**. We'll learn how external factors can influence our ball colors, introducing the concept of covariates and regression into our Bayesian framework.

Think of it as the temperature of the room affecting the color of balls produced - now we have external inputs that influence our generative process!

---

*Continue to [Part 5: Input-Dependent Models and GLMs](../bayesian-tutorial-part5) to learn about incorporating external covariates into your Bayesian models.*

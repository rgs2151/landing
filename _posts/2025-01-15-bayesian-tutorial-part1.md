---
layout: default
title: "Bayesian Modeling - Part 1: From Balls in Bags to Probability"
date: 2025-01-15
tag: thoughts
---

Welcome to the first part of our comprehensive Bayesian modeling tutorial! We'll start with the fundamentals and work our way up to complex models. Think of this as your journey from understanding basic probability to building sophisticated inference systems.

## The Story of Balls in Bags

### 1.1 Probability and Quantifying Variability

Probability is defined as a number between 0 and 1, which describes the likelihood of the occurrence of some particular event in some range of events. 0 means an infinitely unlikely event, and 1 means the certain event. The term 'event' is very general, but for us can usually be thought of as one sample data point in an experiment, and the 'range of events' would be all the data we would get if we sampled virtually an infinite dataset on the experiment.

Imagine a bag with red and blue balls:

*[Image Caption: A transparent bag containing 6 red balls and 6 blue balls, showing equal distribution]*

6 red balls and 6 blue balls.

If you randomly "sample" one ball (a.k.a. event), the probability of getting a red ball is the fraction of red balls to the total.

$$
P(Red)= \frac {\text{Number of Red Balls}​} {\text{Total Number of Balls}}
$$

Similarly, the probability of getting a blue ball is the fraction of blue balls to the total.

$$
P(Blue)= \frac {\text{Number of Blue Balls}​} {\text{Total Number of Balls}}
$$

And the sum of the probabilities of all possible outcomes would be 1.

$$
P(Red) + P(Blue) = 1
$$

### The Generative Process

If you sampled a ball and put it back in the bag, the probability of getting a red ball would be the same on the next draw. However, if you sampled a ball and didn't put it back in the bag, the probability of getting a red ball would change for the next draw.

**For the rest of this tutorial, we will assume that the generative process that makes these balls can infinitely keep making balls and we can draw as much as we want to discover the true distribution.**

With sufficient draws you can be more and more confident about the variability in the probability and the true distribution in the generative process. With a large dataset, the sample probability will converge to the true probability which should be .5/.5 in this case.

Here's a simple demonstration of this convergence:

```python
import numpy as np
import matplotlib.pyplot as plt

# Show the probability converging to .5 .5 with increasing number of samples
n_draws = [5, 10, 100, 1000, 10000]
color = ["#ff4c4c", "#4c4cff"]

fig, ax = plt.subplots(1, 5, figsize=(20, 3))
for i, n in enumerate(n_draws):
    data = np.random.choice(["red", "blue"], n)
    p = np.mean(data == "red")
    var = p * (1 - p) / n
    ax[i].bar(["red", "blue"], [p, 1 - p], color=color)
    ax[i].errorbar(["red", "blue"], [p, 1 - p], yerr=var, fmt="none", ecolor="black", capsize=5)
    ax[i].set_title(f"n = {n}")
    ax[i].set_ylim(0, 1)
```

***Run this code multiple times and see which graph changes most dramatically between runs!***

### 1.2 Updating Beliefs with New Data

As you intuitively understood from the above experiment, as you draw more and more samples, you can be more and more confident about the true distribution of the balls in the bag. This is the **basis of Bayesian statistics**.

We can formalize this process with Bayes' theorem:

$$
P(\text{Red | Data}) = \frac{P(\text{Data | Red}) P(\text{Red})}{P(\text{Data})}
$$

Where:
- $ P(\text{Red}) $: Prior belief about the proportion of red balls.
- $ P(\text{Data | Red}) $: Likelihood of observing the data given the proportion of red balls.
- $ P(\text{Data}) $: Total probability of observing the data.

So, basically, as you acquired more data, you updated your belief about the proportion of red balls in the bag (which in this case is the same as the probability).

### 1.3 From Discrete to Continuous Space

**Notice** how you could either get a red ball or a blue ball. This means your **choices are discrete.**

Sometimes, it is possible that the generative process is **continuous**. Such as, what if the bag had balls in a **spectrum of colors** between red and blue and till now you were only sampling from a specific subset of the dataset. And now for some unknown reason, the generative process has changed and you are now sampling from the whole dataset.

*[Image Caption: A bag showing balls in a continuous spectrum from deep blue through purple to deep red, representing continuous color variation]*

To discover this new true distribution of the generative process, we make histograms of the samples we draw. By making the buckets in the histograms thinner and thinner (to infinity), we can get a better idea of the ***true continuous distribution***.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.colors as mcolors

def get_color_gradient(n, color1="red", color2="blue"):
    cmap = mcolors.LinearSegmentedColormap.from_list("gradient", [color1, color2])
    return [cmap(i / (n - 1)) for i in range(n)]

# Normal distribution parameters
mu, sigma = 0, 1
num_bins_list = [2, 5, 10, 26]

def create_pmf(num_bins, mu, sigma):
    bins = np.linspace(-3, 3, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    pmf_values = norm.pdf(bin_centers, mu, sigma)
    return bins, bin_centers, pmf_values

def plot_approximation(num_bins_list, mu, sigma):
    fig, axes = plt.subplots(1, len(num_bins_list), figsize=(20, 3), sharex=False)
    for ax, num_bins in zip(axes, num_bins_list):
        bins, bin_centers, pmf_values = create_pmf(num_bins, mu, sigma)
        gradient_colors = get_color_gradient(num_bins, color1="red", color2="blue")

        ax.bar(bin_centers, pmf_values, width=(bins[1] - bins[0]), 
               color=gradient_colors, edgecolor='black', alpha=0.7)

        for center, value, color in zip(bin_centers, pmf_values, gradient_colors):
            ax.plot(center, value + 0.05, 'o', color=color, markersize=8)

        ax.set_title(f"PMF Approximation with {num_bins} Balls")
        ax.set_ylabel("Probability (Mass Density)")
        ax.set_xticks(bin_centers)
        ax.set_xticklabels([f"{chr(97 + i)}" for i in range(num_bins)])
        ax.set_xlabel("Smaller bin sizes for better Approximations")

plot_approximation(num_bins_list, mu, sigma)
```

Transitioning from a discrete probability to a continuous probability involves shifting from assigning probabilities to distinct, separate outcomes to describing probabilities across a continuum of possibilities.

**We need a way to model this continuous distribution!**

To do this, we assume that the histogram is an approximation of a continuous function which describes a probability distribution. And we map our histogram to the "best suited" continuous function hoping that this is the true distribution of the generative process.

Below are some examples of different continuous functions that can be used to approximate the histogram:

```python
from scipy.stats import norm, poisson, expon, uniform, halfnorm, beta

x_values = {
    "Normal": np.linspace(-4, 4, 1000),
    "Poisson": np.arange(0, 10, 0.1),
    "Exponential": np.linspace(0, 8, 1000),
    "Uniform": np.linspace(-0.5, 1.5, 1000),
    "Half-Normal": np.linspace(0, 4, 1000),
    "Beta": np.linspace(0, 1, 1000),
}

pdfs = {
    "Normal": norm.pdf(x_values["Normal"], loc=0, scale=1),
    "Poisson": poisson.pmf(x_values["Poisson"].astype(int), mu=3),
    "Exponential": expon.pdf(x_values["Exponential"], scale=1),
    "Uniform": uniform.pdf(x_values["Uniform"], loc=0, scale=1),
    "Half-Normal": halfnorm.pdf(x_values["Half-Normal"]),
    "Beta": beta.pdf(x_values["Beta"], a=.5, b=.5),
}

plt.figure(figsize=(10, 5))
colors = get_color_gradient(len(pdfs))

for i, (dist_name, x_vals) in enumerate(x_values.items()):
    plt.subplot(2, 3, i + 1)
    plt.plot(x_vals, pdfs[dist_name], color=colors[i], label=f"{dist_name} PDF/PMF")
    plt.title(f"{dist_name} Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density / Probability")

plt.show()
```

***So which function should we choose?***

## The Path Forward

*[Image Caption: A flowchart showing the progression from simple stationary models to complex dynamic models, with branches for different types of modeling approaches]*

In this tutorial series, we'll explore:

1. **Stationary Models** (Part 1-2):
   - Single distributions
   - Mixture models
   - Markov models

2. **Dynamic Models** (Part 3-4):
   - Input-dependent distributions (GLMs)
   - Input-dependent mixtures
   - Hidden Markov Models with covariates

Each step builds on the previous one, taking you from basic probability to sophisticated inference systems that can handle complex, real-world data.

## What's Next?

In Part 2, we'll dive into **Bayesian Thinking** and learn how to specify and solve our first Bayesian model. We'll discover how to choose the right distribution and update our beliefs as we see more data.

The journey from balls in bags to advanced Bayesian models is fascinating - let's continue exploring together!

---

*Continue to [Part 2: Bayesian Thinking and Model Specification](../bayesian-tutorial-part2) to learn about priors, likelihoods, and posterior inference.*

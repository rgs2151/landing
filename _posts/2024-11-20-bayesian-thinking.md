---
layout: default
title:  "Tutorial on Bayesian Modeling"
date:   2024-11-23 20:27:20 -0500
---

## 1. Refresher on Probabilities


### 1.1 Probability and Quantifying Variability

Probability is defined as a number between 0 and 1, which describes the likelihood of the occurrence of some particular event in some range of events. 0 means an infinitely unlikely event, and 1 means the certain event. The term ‘event’ is very general, but for us can usually be thought of as one sample data point in an experiment, and the ‘range of events’ would be all the data we would get if we sampled virtually an infinite dataset on the experiment. For now, We are only interested in quantifying the sample variability. [ref](https://geofaculty.uwyo.edu/neil/teaching/Numerical_web/Yaxis_values.pdf)

Imagine a bag with red and blue balls as such: 6 red balls and 6 blue balls.

If you randomly "sample" one ball (a.k.a. event), the probability of getting a red ball is the fraction of red balls to the total.

$$
P(Red)= \frac {\text{Number of Red Balls}​} {\text{Total Number of Balls}}\\
$$

Similarly, the probability of getting a blue ball is the fraction of blue balls to the total.
$$
P(Blue)= \frac {\text{Number of Blue Balls}​} {\text{Total Number of Balls}}\\
$$

And the sum of the probabilities of all possible outcomes would be 1.
$$
P(Red) + P(Blue) = 1
$$

If you sampled a ball and put it back in the bag, the probability of getting a red ball would be the same on the next draw.\
However, if you sampled a ball and didn't put it back in the bag, the probability of getting a red ball would change for the next draw. (It would be less than previous event)\
This is because the sample space has changed (we wont go too much into this for now).

**For the rest of this tutorial, we will assume that the generative process that makes these balls can infinitly keep making balls and we can draw as much as we want to discover the true distribution.**\
If you sampled a ball and put it back in the bag, that is your generative process.\
In this manner, if you repeatedly sample from the bag, there is no limit to the number balls you could draw.\
With sufficient draws you can be more and more confident about the vraiability in the probability and the true distribution in the generative process.\
With a large dataset, the sample probability will converge to the true probability which sould be .5/.5 in this case.

***Run the code cell below multiple times and see which graph changes most dramatically between runs***


### 1.2 Updating beliefs with new data

As you intuitively understood from above experiment, as you draw more and more samples, you can be more and more confident about the true distribution of the balls in the bag.\
This is the **basis of Bayesian statistics**.

We can formalize this process with Bayes' theorem:

$$
P(\text{Red | Data}) = \frac{P(\text{Data | Red}) P(\text{Red})}{P(\text{Data})}
$$

So, basically, as you you aquired more data, you updated your belief about the proportion of red balls in the bag. (which in this case is the same as the probability)

### 1.3 Discrete to Continuous space

**Notice** how you could either get a red ball or either a blue ball. This means your **choices are discrete.**

Sometimes, it is possible that the generative process is **continuous**.\
Such as, what if the bag had balls in a **spectrum of colors** between red and blue and till now you were only sampling from a specific subset of the dataset. And now for some unknown reason, the generative process has changed and you are now sampling from the whole dataset.\


To discover this new true distribution of the generative process, we make histograms of the samples we draw.\
By making the buckets in the histograms thinner and thinner (to infinity), we can get a better idea of the ***true continuous distribution***.

## 2. Bayesian Thinking

### 2.1 Probality Interpretation

Choosing the right distribution is a question of what fits the data best.\
And **when we are aproximating the data, we are fitting a model!**

There are 2 schools of thought on this:  
1. **Frequentist**: They consider the data as fixed (given the experiment) and view the model parameters as unknown but fixed quantities. They use methods like Maximum Likelihood Estimation (MLE) to estimate the parameters that maximize the likelihood of observing the data.

2. **Bayesian**: They consider the data as observed, and the **model parameters as random variables with prior beliefs**. They use **Bayes' theorem to update their prior beliefs** about the parameters into posterior distributions based on the observed data.

**For the purpose of this tutorial, we will focus specifically on the Bayesian approach.**

### 2.2 Bayesian Model Specification

**Bayesian model specification involves 3 components:**

1. **Prior**: The prior distribution describes the beliefs about the model parameters before observing the data. It is often assumed to be a normal distribution.
2. **Likelihood**: The likelihood function describes the probability of observing the data given the model parameters. It is the foundation of the model.
3. **Posterior**: The posterior distribution describes the updated beliefs about the model parameters after observing the data. It is calculated using Bayes' theorem.

We can update our beliefs about the generative process by using the posterior distribution:
$$Posterior = \frac{Likelihood \times Prior}{Evidence}$$


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

### 2.3 Inference

Yayy, now that we have everything, *LETS GO AND SOLVE IT!!!!*\
But wait, how do we solve it?

Solving means calculating the posterior distribution $P(\mu, \sigma | y)$.

- **Analytically**:\
We can solve the posterior distribution analytically for simple models such as this.
- **Numerically**:\
For complex models, it is often **intractable to solve** the posterior distribution analytically. In such cases, we can use numerical methods like **Markov Chain Monte Carlo (MCMC)** or **Variational Inference (VI)** to approximate the posterior distribution.

**For the purpose of this tutorial, we will use MCMC to approximate the posterior distribution.**

## 2. Linear Regressions

### 2.1 Idea of Uncertainty

Traditional linear models are based on the equation:
$$
y = x \cdot w + c
$$
This form represents a **deterministic relationship** between the input $x$ and the output $y$. Every input $x$ maps to a single, exact $y$ without any randomness or uncertainty. 


In real-world data, however, outputs are often affected by factors not captured by the model (e.g., measurement errors, hidden variables, or inherent variability). Adding noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ (for example) acknowledges this uncertainty and makes the model more realistic, leading to:

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

This directly states that $y$ is drawn from a normal distribution with mean $x \cdot w + c$ and variance $\sigma^2$. (A typical Bayesian representation)

The noisy version better reflects the variability seen in observed data.

### 2.2 Model Specification for Bayesian Linear Regression
We can specify our model and its parameters as follows:

**Priors**: 
$$w \sim \mathcal{N}(0, 1)$$ 
$$c \sim \mathcal{N}(0, 1)$$
$$\sigma \sim |\mathcal{N}(0, 1)|$$
The prior distributions are initially chosen to be normal with mean $0$ and variance $1$ for simplicity.

**Likelihood**: 
$$y \sim \mathcal{N}(x \cdot w + c, \sigma^2)$$


**Posterior**:\
We will update those priors based on Bayes' theorem (above):

$$
P(w, c, \sigma | y, x) = \frac{P(y | x, w, c, \sigma) P(w) P(c) P(\sigma)}{P(y)}
$$

Where:
- $P(y | x, w, c, \sigma)$ is the likelihood of observing the data given the parameters.
- $P(w)$, $P(c)$, and $P(\sigma)$ are the prior distributions of the parameters.
- $P(y)$ is the total probability of observing the data.

<!-- 
{% highlight python %}

{% endhighlight %} -->

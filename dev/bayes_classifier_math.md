# Mathematical Reference: Gaussian Mixture Log-Probabilities and Bayes Classification

---

**Disclaimer:** This document was AI generated (Claude AI, [see chat](https://claude.ai/share/a9be6937-d7bb-4574-83bd-5a6f36e4b228)) with few adptations by the packages author.

## 1. Log-Probability of a Multivariate Normal via Cholesky Decomposition

For a single observation $x \in \mathbb{R}^k$ under a multivariate normal $\mathcal{N}(\mu, \Sigma)$, the log-density is:

$$\log p(x) = -\frac{k}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2}(x - \mu)^\top \Sigma^{-1}(x - \mu)$$

**Computing the Mahalanobis term via Cholesky.** The Cholesky decomposition gives $\Sigma = LL^\top$ where the cholesky decomposition $L$ is lower triangular. Then:

$$\Sigma^{-1} = (LL^\top)^{-1} = L^{-\top}L^{-1}$$

so the quadratic form becomes:

$$(x-\mu)^\top \Sigma^{-1}(x-\mu) = (x-\mu)^\top L^{-\top}L^{-1}(x-\mu) = \|L^{-1}(x-\mu)\|^2_2 =: \|v\|^2_2$$

where $v$ solves the triangular system $Lv = (x - \mu)$. Rather than forming $\Sigma^{-1}$ explicitly, we call `torch.linalg.solve_triangular(L, x - μ, upper=False)` which is cheaper and numerically more stable.

**Computing the log-determinant via Cholesky.** Since $\Sigma = LL^\top$:

$$\log|\Sigma| = \log|LL^\top| = 2\log|L|$$

and since $L$ is triangular, its determinant is the product of its diagonal entries:

$$\log|L| = \sum_{i=1}^{k}\log L_{ii}$$

so $\log|\Sigma| = 2\sum_{i}\log L_{ii}$, computed in code as `2 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)`.

**In code**, the per-component log-probability for all $n$ observations and $m$ components simultaneously has shape `(b, n, m)` and is assembled as:

```
log_norm  = -0.5 * (k * log(2π) + log_det)   # (b, m)       — normalisation constant
comp_log_probs = log_norm.unsqueeze(1) - 0.5 * mahal  # (b, n, m)
```

**For a GMM**, the log-density is $\log\sum_j w_j \mathcal{N}(x \mid \mu_j, \Sigma_j)$, evaluated stably via log-sum-exp:

$$\log p(x) = \text{logsumexp}_j\bigl(\log w_j + \log\mathcal{N}(x\mid\mu_j,\Sigma_j)\bigr)$$

> **Implementation note.** A naive summation $\sum_j w_j \mathcal{N}(\cdot)$ in linear space is avoided because individual Gaussian densities can underflow to zero for large $k$. Working entirely in log-space with `torch.logsumexp` prevents this.

---

## 2. Effect of Additive White Noise on the Log-Probability

If an observation is generated as:

$$x = z + \varepsilon, \quad z \sim \text{GMM}, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

then by the convolution property of Gaussians, each component $\mathcal{N}(\mu_j, \Sigma_j)$ becomes $\mathcal{N}(\mu_j, \Sigma_j + \sigma^2 I)$. The mixture weights and means are unchanged. The inflated covariance is formed as:

$$\Sigma_j^{\text{noisy}} = LL^\top + \sigma^2 I$$

and a fresh Cholesky decomposition $\Sigma_j^{\text{noisy}} = \tilde{L}\tilde{L}^\top$ is computed on the inflated matrix. All subsequent log-prob computations then proceed with $\tilde{L}$ in place of $L$, including the log-determinant term which must also use $\tilde{L}$.

> **Note.** In code this is controlled by the `white_noise_var` argument to `log_prob`. When `white_noise_var = 0` the original Cholesky factor is used directly; otherwise a new Cholesky is computed on the inflated covariance and used for both the Mahalanobis solve and the log-determinant.

---

## 3. Bayes-Optimal Classifier

Given a new observation $x$, let $\rho = P(\text{bad})$ be the prior probability of the bad class. By Bayes' theorem the posterior is:

$$P(\text{bad} \mid x) = \frac{\rho\, p_{\text{bad}}(x)}{\rho\, p_{\text{bad}}(x) + (1-\rho)\, p_{\text{good}}(x)}$$

The **Bayes-optimal classifier** minimises the probability of error and decides:

$$\hat{y}(x) = \begin{cases} \text{bad} & \text{if } P(\text{bad}\mid x) > 0.5 \\ \text{good} & \text{otherwise} \end{cases}$$

which is equivalently expressed as a threshold on the **log-likelihood ratio**:

$$\hat{y}(x) = \text{bad} \iff \log\frac{p_{\text{bad}}(x)}{p_{\text{good}}(x)} > \log\frac{1-\rho}{\rho}$$

**In code**, the posterior is never computed in linear space. Instead we work with the log-weighted densities:

```
log_rho_p_bad  = log(ρ)   + log_p_bad    # log[ ρ · p_bad(x) ]
log_rho_p_good = log(1-ρ) + log_p_good   # log[ (1-ρ) · p_good(x) ]
```

The classifier decision reduces to comparing these two scalars: `pred_bad = log_rho_p_bad > log_rho_p_good`, which avoids computing the normalising constant altogether for the decision step.

The posterior itself, needed for `confusion_probability`, is computed via:

```
log_marginal  = logaddexp(log_rho_p_bad, log_rho_p_good)
posterior_bad = exp(log_rho_p_bad - log_marginal)
```

> **Implementation note.** The marginal $\rho p_{\text{bad}}(x) + (1-\rho)p_{\text{good}}(x)$ would naively be computed in linear space, but `torch.logaddexp` computes $\log(e^a + e^b)$ stably from $a = \log(\rho p_{\text{bad}})$ and $b = \log((1-\rho) p_{\text{good}})$, preventing overflow/underflow when densities are very small or very large.

---

## 4. Bayes Error Rate

The **Bayes error rate** $\epsilon^*$ is the minimum achievable probability of misclassification, attained by the Bayes-optimal classifier:

$$\epsilon^* = \int \min\bigl(\rho\, p_{\text{bad}}(x),\;(1-\rho)\, p_{\text{good}}(x)\bigr)\, dx$$

Re-writing as an expectation under the marginal $p(x) = \rho\, p_{\text{bad}}(x) + (1-\rho)\, p_{\text{good}}(x)$:

$$\epsilon^* = \mathbb{E}_{x \sim p}\!\left[\frac{\min\bigl(\rho\, p_{\text{bad}}(x),\;(1-\rho)\, p_{\text{good}}(x)\bigr)}{p(x)}\right] = \mathbb{E}_{x\sim p}\!\Bigl[\min\bigl(P(\text{bad}\mid x),\,P(\text{good}\mid x)\bigr)\Bigr]$$

This simplifies further: the minimum of the two posteriors is the probability assigned to the **wrong** class, which is exactly 1 when the classifier errs and 0 otherwise — so $\epsilon^*$ is simply the probability of a mistake under the joint:

$$\epsilon^* = P_{(x,y)\sim p_{\text{joint}}}\!\bigl(\hat{y}(x) \neq y\bigr)$$

**Monte Carlo estimator.** Draw $N$ joint samples $(x^{(i)}, y^{(i)})$ from the full DGP (i.e. from the marginal with true labels attached), apply the Bayes classifier, and count errors:

$$\hat{\epsilon}^* = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\bigl[\hat{y}(x^{(i)}) \neq y^{(i)}\bigr]$$

This is an unbiased estimator of $\epsilon^*$ because sampling from the joint $p(x,y)$ is equivalent to sampling $x$ from the marginal — the true label $y^{(i)}$ is used only for the error indicator, not in the classifier decision.

**In code:**

```
pred_bad  = log_rho_p_bad > log_rho_p_good   # Bayes decision
true_bad  = y == bad_encoding                 # ground truth
errors    = pred_bad XOR true_bad             # wrong iff exactly one is True
ε̂  = errors.float().mean()
```

**Default sample size.** To ensure sufficient coverage of the mixture geometry the default is $N = \max(10^4,\; 100 \cdot m \cdot k)$, scaling with the number of components $m$ and the dimension $k$.

---

## 5. Confusion Probability

**Note:** This confusion probability was thought of by the author, the mathematical formulation and precisement was done with AI.

For a threshold pair $0 < d_{\text{lower}}, d_{\text{upper}} < 0.5$, the **confusion probability** is the marginal probability mass of the input space that falls inside the confusion band around the decision boundary:

$$\Pi(d_{\text{lower}}, d_{\text{upper}}, \rho) = P_{x \sim p}\!\Bigl(P(\text{bad}\mid x) \in [0.5 - d_{\text{lower}},\; 0.5 + d_{\text{upper}}]\Bigr)$$

Intuitively, this is the fraction of observations for which even a perfect classifier is uncertain. When $d_{\text{lower}} = d_{\text{upper}} = d$ the band is symmetric. Asymmetric bands are meaningful from a risk perspective. E. g. financial institutions typically operate at thresholds well above 0.5, so the operationally relevant confusion region is not symmetric around the statistical decision boundary.

As the band widens, $\Pi \to 1$. As the band shrinks, $\Pi \to 0$ for well-separated classes and $\Pi \to 1$ for fully overlapping classes. The relationship to the Bayes error rate is:

$$\lim_{d_{\text{lower}}, d_{\text{upper}} \to 0} \Pi(d_{\text{lower}}, d_{\text{upper}}, \rho) \;\to\; 0$$

while a large $\Pi$ at small $d$ is the clearest indicator of a hard classification problem.

**Monte Carlo estimator.** Draw $N$ samples $x^{(i)}$ from the marginal, compute the posterior for each, and estimate the band probability:

$$\hat{\Pi}(d_{\text{lower}}, d_{\text{upper}}, \rho) = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\!\left[P(\text{bad}\mid x^{(i)}) \in [0.5 - d_{\text{lower}},\; 0.5 + d_{\text{upper}}]\right]$$

**In code:**

```
log_marginal  = logaddexp(log_rho_p_bad, log_rho_p_good)   # stable log-sum
posterior_bad = exp(log_rho_p_bad - log_marginal)           # P(bad | x)
in_band       = (posterior_bad >= 0.5 - d_lower) & (posterior_bad <= 0.5 + d_upper)
Π̂  = in_band.float().mean()
```

> **Implementation note.** The posterior is obtained by exponentiating the difference of log-quantities rather than dividing in linear space, which again avoids numerical issues when densities are very small.

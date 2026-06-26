"""
Mathematical and Gaussian Helper Functions

Provides statistical tools including probability density, cumulative distribution,
inverse normal distribution approximations, and the Shapiro-Wilk test for normality.
"""

import math
import numpy as np

def norm_pdf(x, mu, sigma):
    """Compute the normal probability density function (PDF)."""
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * \
           np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def norm_fit(data):
    """Estimate mean and standard deviation of data (degrees of freedom = 1)."""
    return float(np.mean(data)), float(np.std(data, ddof=1))

def _inv_norm(p):
    """Compute the inverse cumulative normal distribution using Hastings' rational approximation."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
        return -(t - (2.515517 + 0.802853 * t + 0.010328 * t ** 2) /
                 (1.0 + 1.432788 * t + 0.189269 * t ** 2 + 0.001308 * t ** 3))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        return t - (2.515517 + 0.802853 * t + 0.010328 * t ** 2) / \
               (1.0 + 1.432788 * t + 0.189269 * t ** 2 + 0.001308 * t ** 3)

def _erf(x):
    """Compute the error function using polynomial approximation."""
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592
                + t * (-0.284496736
                       + t * (1.421413741
                              + t * (-1.453152027
                                     + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-ax * ax))

def norm_cdf(x):
    """Compute the standard normal cumulative distribution function (CDF)."""
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))

def shapiro_wilk_p(data):
    """
    Perform Shapiro-Wilk test for normality using Royston's log-transform approximation.
    Supports sample sizes from 3 up to 5000 (random sampling is applied for larger datasets).
    """
    data = np.asarray(data, dtype=float)
    if len(data) > 5000:
        rng = np.random.default_rng(42)
        data = rng.choice(data, 5000, replace=False)
    data = np.sort(data)
    n = len(data)
    if n < 3:
        return float("nan")
    mi = np.array([_inv_norm((i - 0.375) / (n + 0.25)) for i in range(1, n + 1)])
    mi_norm = math.sqrt(float(np.dot(mi, mi)))
    a = mi / mi_norm
    xbar = float(data.mean())
    numerator = float(np.dot(a, data)) ** 2
    denominator = float(np.sum((data - xbar) ** 2))
    W = min(max(numerator / denominator if denominator > 0 else 1.0, 1e-10), 1.0 - 1e-10)
    ln_n = math.log(n)
    lnW = math.log(1.0 - W)
    if n >= 12:
        mu_w = -1.2725 + 1.0521 * ln_n
        sig_w = math.exp(-1.2185 + 1.1883 * ln_n) ** 0.5
    else:
        mu_w = -0.0006714 * n ** 3 + 0.025054 * n ** 2 - 0.39978 * n + 0.5441
        sig_w = math.exp(-0.0020322 * n ** 3 + 0.062767 * n ** 2 - 0.77857 * n + 1.3822)
    z = (lnW - mu_w) / max(sig_w, 1e-9)
    p = float(1.0 - norm_cdf(z))
    return max(0.0, min(1.0, p))

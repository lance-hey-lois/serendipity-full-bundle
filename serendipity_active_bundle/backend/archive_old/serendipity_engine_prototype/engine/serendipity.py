
import math
import random

def make_bins(n=5):
    # novelty bins in [0,1)
    return [{"lo": i/n, "hi": (i+1)/n, "alpha": 1.0, "beta": 1.0} for i in range(n)]

def _beta_like_sample(alpha, beta):
    # simple mean+noise approximation for demo (okay for prototype)
    mean = alpha / (alpha + beta)
    noise = (random.random() - 0.5) / ((alpha + beta + 1)**0.5)
    return max(0.0, min(1.0, mean + noise))

def pick_bin(bins):
    best = -1.0
    idx = 0
    for i, b in enumerate(bins):
        s = _beta_like_sample(b["alpha"], b["beta"])
        if s > best:
            best = s
            idx = i
    return idx

def update_bin(b, reward):
    # reward in [0,1]
    b["alpha"] += reward
    b["beta"]  += (1.0 - reward)

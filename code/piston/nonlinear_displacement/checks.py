import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def Lhat(t, delta, omega):
    fs = 1.0 - delta * (1 - np.cos(omega * t))
    return fs


def F(y, params):

    mu = params.get("mu")
    sigma = params.get("sigma")
    p = params.get("p")

    nonlinear = p * np.exp(-(((y - mu) / sigma) ** 2.0))
    return nonlinear


NX = 1000
T = 0.5

DELTA = 0.15
OMEGA = 2.0 * np.pi * 1

params = {"mu": 0.5, "sigma": 0.1, "p": 0.25}

X = np.linspace(0, 1, NX)

distortion = F(X, params)
displacement = X * (1 + distortion) * (Lhat(T, DELTA, OMEGA) - 1)

x = X + displacement

fig, (top, bottom) = plt.subplots(nrows=2)
top.plot(X, x)
top.plot([0, 1.0], [0, 1.0], linestyle="--")
top.plot(X, X * (Lhat(T, DELTA, OMEGA) - 1) * distortion, linestyle=":")
top.plot(X, X * (Lhat(T, DELTA, OMEGA)), linestyle="-.")
top.set_xlabel("$\\mathcal{X}$")
top.set_ylabel("$x$")
top.grid(True)

Xm = (X[1:] + X[:-1]) / 2
delta_x = np.diff(x)
delta_X = np.diff(X)

bottom.semilogy(Xm, delta_X, linestyle="--")
bottom.semilogy(Xm, delta_x)
bottom.set_xlabel("$\\mathcal{X}_m$")
bottom.set_ylabel("$\\Delta x$")
bottom.grid(True)
plt.show()
plt.close()

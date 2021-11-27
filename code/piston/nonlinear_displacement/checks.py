import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from romtime.conventions import FIG_KWARGS

sns.set_theme(context="paper")

NX = 1000
DELTA = 0.3
TYPE = "compression"
# TYPE = "expansion"


def Lhat(delta):

    if TYPE == "compression":
        fs = 1.0 - delta
    elif TYPE == "expansion":
        fs = 1.0 + delta

    return fs


def F(y, params):

    mu = params.get("mu")
    sigma = params.get("sigma")
    p = params.get("p")

    nonlinear = p * np.exp(-(((y - mu) / sigma) ** 2.0))
    return nonlinear


params = {"mu": 0.5, "sigma": 0.1, "p": 0.75}

X = np.linspace(0, 1, NX)

distortion = F(X, params)
displacement = X * (1 + distortion) * (Lhat(DELTA) - 1)

x = X + displacement

is_feasible = np.all(np.diff(x) > 0)

# -----------------------------------------------------------------------------
# Visualization
fig, (top, bottom) = plt.subplots(nrows=2, gridspec_kw={"hspace": 0.3})
top.plot(X, x)
top.plot([0, 1.0], [0, 1.0], alpha=0.5, c="grey", label="Initial configuration")
top.plot(
    X,
    X * (Lhat(DELTA) - 1) * distortion,
    linestyle=":",
    label="Node concentration",
)
top.plot(X, X * (Lhat(DELTA)), linestyle="-.", label="Uniform distribution")
top.set_xlabel("$\\mathcal{X}$")
top.set_ylabel("$x$")
top.grid(True)
top.legend()
top.set_title(f"Mesh transformation for {TYPE.lower()}")

Xm = (X[1:] + X[:-1]) / 2
delta_x = np.diff(x)
delta_X = np.diff(X)

bottom.semilogy(Xm, delta_X * Lhat(DELTA), linestyle="--", label="Uniform")
bottom.semilogy(Xm, delta_x, label="Nonlinear")
bottom.set_xlabel("$\\mathcal{X}_m$")
bottom.set_ylabel("$\\Delta x$")
bottom.grid(True)
bottom.legend()
bottom.set_ylim([1e-6, 1e-1])


mu = params.get("mu")
sigma = params.get("sigma")
p = params.get("p")
plt.savefig(
    f"mapping_mu_{mu}_sigma_{sigma}_p_{p}_{TYPE}_delta_{DELTA}.png", **FIG_KWARGS
)
plt.close()
import matplotlib.pyplot as plt
import pickle
import ujson
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import more_itertools as mr
from romtime.conventions import RomParameters, Stage, FIG_KWARGS
from romtime.fom.nonlinear import OneDimensionalBurgersConventions as Parameters

here = Path(__file__)
PATH_MU_SPACE = Path(
    here.parents[2]
    / "results"
    / "piston"
    / "hrom_tol_mu"
    / "benchmark"
    / "mu_space.json"
)

PATH_MU_SPACE_DEIM = Path(
    here.parents[2]
    / "results"
    / "piston"
    / "hrom_tol_mu"
    / "tol05_rb_nonlinear"
    / "mu_space_deim.json"
)

PATH_ERRORS_NONLINEAR = Path(
    here.parents[2]
    / "results"
    / "piston"
    / "hrom_tol_mu"
    / "tol05_rb"
    / "errors_deim_nonlinear.csv"
)
PATH_REDUCED_BASIS = Path(
    here.parents[2] / "results" / "piston" / "hrom_tol_mu" / "tol05_rb" / "rb_basis.pkl"
)


def plot_mu_space():

    with PATH_MU_SPACE.open(mode="r") as fp:
        _mu_space = ujson.load(fp)

    mu_space = pd.DataFrame()

    for stage, data in _mu_space.items():

        if stage == Stage.VALIDATION:
            continue

        _data = pd.DataFrame(data)

        _data = _data.drop([Parameters.GAMMA, Parameters.ALPHA], axis=1)
        _data["Stage"] = stage
        mu_space = mu_space.append(_data)

    mu_space = mu_space.reset_index(drop=True)

    g = sns.pairplot(data=mu_space, hue="Stage")
    plt.savefig("mu_space.png", **FIG_KWARGS)


def plot_mu_space_deim():

    with PATH_MU_SPACE_DEIM.open(mode="r") as fp:
        _mu_space = ujson.load(fp)

    mu_space = pd.DataFrame()

    for operator, space in _mu_space.items():

        for stage, data in space.items():

            if stage == Stage.VALIDATION:
                continue

            _data = pd.DataFrame(data)

            _data = _data.drop([Parameters.GAMMA, Parameters.ALPHA], axis=1)
            _data["Stage"] = stage
            _data["Operator"] = operator
            mu_space = mu_space.append(_data)

    mu_space = mu_space.reset_index(drop=True)

    g = sns.PairGrid(mu_space, hue="Operator")
    g.map_offdiag(sns.scatterplot, size=mu_space["Stage"])
    g.add_legend(title="", adjust_subtitles=True)
    plt.savefig("mu_space_deim.png", **FIG_KWARGS)


def plot_nonlinear_errors():
    T = 1.0
    nt = 500
    ts = np.linspace(T / nt, T, nt // 4)
    data = pd.read_csv(PATH_ERRORS_NONLINEAR, index_col=0)

    data.index = ts

    data.plot(legend=False, grid=True)
    plt.title("MDEIM Nonlinear Online Errors")
    plt.xlabel("t (s)")
    plt.ylabel("L2 Error")
    plt.savefig("mdeim_nonlinear_full.png", **FIG_KWARGS)


# plot_mu_space()
# plot_mu_space_deim()
# plot_nonlinear_errors()
with PATH_REDUCED_BASIS.open(mode="rb") as fp:
    basis = pickle.load(fp)

NX = 1000
x = np.linspace(0.0, 1.0, NX + 1)

ns = basis.shape[1]
print(ns)

size = 6
chunks = mr.chunked(range(ns), size)
chunks = list(chunks)

nrows = len(chunks)
fig, axes = plt.subplots(
    nrows=nrows, ncols=1, gridspec_kw={"hspace": 0.2}, sharex=True, sharey=False
)

axes = axes.flatten()
for idx_ax, chunk_ax in enumerate(zip(chunks, axes)):

    chunk, ax = chunk_ax
    for idx_basis in chunk:
        element = basis[:, idx_basis]
        element = np.flip(element)
        ax.plot(x, element)
    ax.grid(True)

    if idx_ax == 0:
        ax.set_title("Reduced Basis Elements (Half-Space)")
    if idx_ax == (nrows - 1):
        ax.set_xlabel("$x$")

plt.savefig("reduced_basis.png", **FIG_KWARGS)
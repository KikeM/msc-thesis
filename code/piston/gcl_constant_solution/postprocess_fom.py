import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle

from romtime.conventions import FIG_KWARGS

from pathlib import Path

sns.set_theme(context="paper")

fixed_paths = Path(".").glob(pattern="*fixed.pkl")
moving_paths = Path(".").glob(pattern="*moving.pkl")

func = lambda x: x.stem.split("_")[-1]
fixed_paths = sorted(fixed_paths, key=func)
moving_paths = sorted(moving_paths, key=func)

COLS = ["fixed", "moving"]
FIXED = COLS[0]
MOVING = COLS[1]

ts = np.linspace(0, 0.75, int(5e2))
results_fixed = pd.DataFrame(index=ts)
results_moving = pd.DataFrame(index=ts)
NXs = [5e1, 1e2, 2e2, 3e2, 5e2, 1e3]
# NXs = [5e1, 1e2, 2e2]
for fixed, moving, nx in zip(fixed_paths, moving_paths, NXs):

    with open(fixed, mode="rb") as fp:
        fom_fixed = pickle.load(fp)
    with open(moving, mode="rb") as fp:
        fom_moving = pickle.load(fp)

    mean_fixed = np.mean(fom_fixed.fom, axis=0)
    mean_moving = np.mean(fom_moving.fom, axis=0)

    results_fixed[nx] = np.abs(mean_fixed)
    results_moving[nx] = np.abs(mean_moving)

results_moving = results_moving.clip(lower=0.1, upper=10)

fig, (top, middle, bottom) = plt.subplots(
    nrows=3, sharex=True, sharey=False, gridspec_kw={"hspace": 0.35}
)

for nx in NXs:
    nx = int(nx)
    top.semilogy(
        results_fixed.index, results_fixed[nx], label=f"$N_x$ = {nx}", linestyle="--"
    )

for nx in NXs[:3]:
    nx = int(nx)
    middle.semilogy(results_moving.index, results_moving[nx], label=f"$N_x$ = {nx}")

for nx in NXs:
    nx = int(nx)
    bottom.semilogy(
        results_moving.index, results_moving[nx], label=f"$N_x$ = {nx}", alpha=0.65
    )

top.set_ylim(top=1.01)

top.set_ylabel("Mean (FE Vector)")
bottom.set_xlabel("t (s)")

top.grid(True)
bottom.grid(True)

top.set_title("Fixed Domain")
middle.set_title("Moving Domain (First $N_x$)")
bottom.set_title("Moving Domain")

bottom.legend(ncol=3)
middle.legend()
top.legend(ncol=3)

# plt.show()
plt.savefig("mean_fe_comparison_constant_solution.png", **FIG_KWARGS)
plt.close()

results_fixed.to_csv("results_fixed.csv")
results_moving.to_csv("results_moving.csv")

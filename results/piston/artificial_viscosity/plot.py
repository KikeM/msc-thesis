import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from romtime.conventions import FIG_KWARGS

from pathlib import Path

sns.set_theme(context="paper")

path_probes = Path(".").glob("*.csv")
is_probe = lambda x: "probe" in x.stem
path_probes = filter(is_probe, path_probes)
path_probes = sorted(path_probes, key=lambda y: int(y.stem.split("_")[-1]))

alphas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]

outflows = pd.DataFrame()
for path, alpha in zip(path_probes, alphas):

    df = pd.read_csv(path, index_col=0)
    outflow = df["0.0"].squeeze()

    name = path.stem.split("_")[-1]
    outflow.name = "%.1e" % alpha

    outflows = pd.concat([outflows, outflow], axis=1)

outflows.columns = outflows.columns.astype(str)

print(outflows.columns)


piston = df["L"].squeeze()
piston.name = "piston"

# Include initial state
outflows.loc[0.0] = 0.0
piston.loc[0.0] = 0.0
outflows = outflows.sort_index()
piston = piston.sort_index()

fig, (top, bottom) = plt.subplots(nrows=2, gridspec_kw={"hspace": 0.5})

damping_label = "$\\varepsilon \\sim 10^{-6}$"
convective_label = "$\\varepsilon \\sim 10^{-10}$"

top.plot(outflows.index, outflows["1.0e-06"], label=damping_label)
top.plot(outflows.index, outflows["1.0e-10"], label=convective_label)
top.plot(piston.index, piston.values, linestyle="--", label="Piston", alpha=0.75)
top.set_xlabel("$t$ (s)")
top.set_ylabel("$u$ (m/s)")
top.set_title("Artificial Viscosity Comparison")
top.legend()

# outflows = pd.concat([outflows, piston], axis=1)

# correlations = outflows.corr()
# sns.heatmap(correlations, annot=True)
# plt.savefig("correlations.png", **FIG_KWARGS)
# plt.close()

sns.scatterplot(
    piston.values,
    outflows["1.0e-06"],
    legend=False,
    label=damping_label,
    ax=bottom,
    size=5,
    alpha=0.5,
)
sns.scatterplot(
    piston.values,
    outflows["1.0e-10"],
    legend=False,
    label=convective_label,
    ax=bottom,
    marker="s",
    size=5,
    alpha=0.75,
)
# bottom.legend()
bottom.set_xlabel("Piston (m/s)")
bottom.set_ylabel("Outflow (m/s)")
bottom.set_title("Phase Plot")

# plt.show()
plt.savefig("artificial_viscosity_comparison.png", **FIG_KWARGS)
plt.close()
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

piston = df["L"].squeeze()
piston.name = "piston"


ax = outflows.plot(grid=True)
ax.plot(piston.index, piston.values, linestyle="--", label="piston", alpha=0.75)
ax.set_xlabel("$t$ (s)")
ax.set_ylabel("$u$ (m/s)")
ax.set_title("Artificial Viscosity Comparison")
plt.savefig("timeseries.png", **FIG_KWARGS)
plt.close()

outflows = pd.concat([outflows, piston], axis=1)

correlations = outflows.corr()
sns.heatmap(correlations, annot=True)
plt.savefig("correlations.png", **FIG_KWARGS)
plt.close()

sns.pairplot(outflows)
plt.savefig("pairplot.png", **FIG_KWARGS)
plt.close()

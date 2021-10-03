import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from romtime.conventions import FIG_KWARGS, MassConservation

from pathlib import Path

sns.set_theme(context="paper")


def compute_mass_deficit(mass_data):

    T0 = 0.1
    mask = mass_data[MassConservation.TIMESTEPS] > T0
    mass_data = mass_data[mask]

    mass_change = mass_data[MassConservation.MASS_CHANGE]
    outflow = mass_data[MassConservation.OUTFLOW]

    md = mass_change - outflow
    md = np.abs(md)
    md = np.mean(md)

    # N = len(md)
    # md = np.linalg.norm(md)
    # md /= np.sqrt(N)

    return md


path_mass = Path(".").glob("mass*.csv")
is_bdf_one = lambda x: "bdf_1" in x.stem
path_bdf_one = filter(is_bdf_one, path_mass)

path_mass = Path(".").glob("mass*.csv")
is_bdf_two = lambda x: "bdf_2" in x.stem
path_bdf_two = filter(is_bdf_two, path_mass)

path_bdf_one = sorted(path_bdf_one, key=lambda y: int(y.stem.split("_")[2]))
path_bdf_two = sorted(path_bdf_two, key=lambda y: int(y.stem.split("_")[2]))

nts = [5e1, 1e2, 3e2, 5e2, 1e3, 2e3, 3e3, 5e3, 1e4]

COLUMNS = ["BDF-1", "BDF-2", "identity"]
BDF1 = COLUMNS[0]
BDF2 = COLUMNS[1]
IDENTITY = COLUMNS[2]


results = pd.DataFrame(columns=COLUMNS, dtype=np.float32)
T = 0.75
for bdf_1, bdf_2, nt in zip(path_bdf_one, path_bdf_two, nts):

    mass_data_1 = pd.read_csv(bdf_1, index_col=0)
    mass_data_2 = pd.read_csv(bdf_2, index_col=0)

    md_1 = compute_mass_deficit(mass_data_1)
    md_2 = compute_mass_deficit(mass_data_2)

    dt = T / nt

    results.loc[dt, BDF1] = md_1
    results.loc[dt, BDF2] = md_2
    results.loc[dt, IDENTITY] = dt

slope_1, _ = np.polyfit(
    x=np.log10(results.index[1:]), y=np.log10(results[BDF1].values[1:]), deg=1
)
slope_2, _ = np.polyfit(
    x=np.log10(results.index[1:]), y=np.log10(results[BDF2].values[1:]), deg=1
)

print(f"BDF-1 (mass defect rate): {-np.round(slope_1,2)}")
print(f"BDF-2 (mass defect rate): {-np.round(slope_2,2)}")

options = dict(linestyle="--", marker=".", markersize=8)
plt.loglog(
    results.index,
    results[BDF1],
    **options,
    label=BDF1 + f" rate: {-np.round(slope_1,2)}",
)
plt.loglog(
    results.index,
    results[BDF2],
    **options,
    label=BDF2 + f" rate: {-np.round(slope_2,2)}",
)
plt.axis("tight")
# plt.loglog(
#     results.index,
#     results[IDENTITY],
#     label=IDENTITY,
#     linestyle="--",
#     alpha=0.5,
#     color="black",
# )
plt.legend()
plt.grid(True)
plt.ylim(bottom=1e-3, top=1.0)
plt.xlabel("$dt$")
plt.ylabel("Mean Error")
plt.title("Mass Defect - Convergence Rates")
plt.savefig("convergence_rates_mass.png", **FIG_KWARGS)
plt.close()


# outflows = pd.DataFrame()

#     df = pd.read_csv(path, index_col=0)
#     outflow = df["0.0"].squeeze()

#     name = path.stem.split("_")[-1]
#     outflow.name = "%.1e" % alpha

#     outflows = pd.concat([outflows, outflow], axis=1)

# piston = df["L"].squeeze()
# piston.name = "piston"


# ax = outflows.plot(grid=True)
# ax.plot(piston.index, piston.values, linestyle="--", label="piston", alpha=0.75)
# ax.set_xlabel("$t$ (s)")
# ax.set_ylabel("$u$ (m/s)")
# ax.set_title("Artificial Viscosity Comparison")
# plt.savefig("timeseries.png", **FIG_KWARGS)
# plt.close()

# outflows = pd.concat([outflows, piston], axis=1)

# correlations = outflows.corr()
# sns.heatmap(correlations, annot=True)
# plt.savefig("correlations.png", **FIG_KWARGS)
# plt.close()

# sns.pairplot(outflows)
# plt.savefig("pairplot.png", **FIG_KWARGS)
# plt.close()

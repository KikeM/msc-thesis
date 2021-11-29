from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from romtime.conventions import FIG_KWARGS, Stage, Treewalk
from tqdm import tqdm
import numpy as np

sns.set_theme(context="paper", palette="colorblind")

# SNS_SET = "Paired"
SNS_SET = "Set1"

# # -----------------------------------------------------------------------------
# # Parameter Space

# mu_space_offline = pd.read_json("error_evaluation_mu_space_offline.json")
# mu_space_online = pd.read_json("error_evaluation_mu_space_online.json")

# mu_space_offline["Stage"] = Stage.OFFLINE.capitalize()
# mu_space_online["Stage"] = Stage.ONLINE.capitalize()

# mu_space = pd.concat([mu_space_offline, mu_space_online], axis=0)
# mu_space = mu_space.reset_index(drop=True)

# # Enforce order
# mu_space = mu_space[["a0", "delta", "omega", "piston_mach", "loc", "scale", "sigma", "Stage"]]
# mu_space = mu_space.rename(
#     columns={
#         "a0": "$a_0$",
#         "delta": "$\\delta$",
#         "omega": "$\\omega$",
#         "piston_mach": "$u_p$",
#         "loc": "$x_c$",
#         "scale": "$y_c$",
#         "sigma": "$\\sigma_c$",
#     }
# )

# vars = list(mu_space.columns)
# vars.remove("Stage")

# sns.pairplot(mu_space, hue="Stage", vars=vars, diag_kind="hist")
# plt.show()
# plt.close()

# # -----------------------------------------------------------------------------
# # Operators error decay
# print("Error Decay")
# FILES_ERRORS_ONLINE = list(Path("online/").glob("errors_deim*.csv"))
# FILES_ERRORS_OFFLINE = list(Path("offline/").glob("errors_deim*.csv"))

# fig, (ax_off, ax_on) = plt.subplots(nrows=2, sharex=True)

# for ax, stage, files in [
#     (ax_off, "offline", FILES_ERRORS_OFFLINE),
#     (ax_on, "online", FILES_ERRORS_ONLINE),
# ]:

#     operators = []
#     errors = pd.DataFrame(columns=["0", "1", "2", "3", "operator"])
#     for file in files:

#         stem = file.stem.split("_")

#         p = stem[-1]
#         if p == "None":
#             p = 1.0
#         else:
#             p = float(p)
#         operator = stem[2]

#         operators.append(operator)

#         # -------------------------------------------------------------------------
#         # Time integration
#         _errors = pd.read_csv(file, index_col=0)
#         _errors.index.name = "ts"
#         _errors = _errors.mean(axis=0)
#         _errors.name = p
#         _errors = pd.DataFrame(_errors).T
#         _errors["operator"] = operator

#         errors = pd.concat([errors, _errors], axis=0)

#     errors = errors.sort_values(by="operator")
#     operators = list(set(operators))
#     operators.append("reduced-basis")
#     operators = sorted(operators)

#     print(operators)

#     # # Use default seaborn color palette
#     # # https://www.codecademy.com/articles/seaborn-design-ii
#     num_colors = len(operators)
#     color = sns.color_palette(palette=SNS_SET, n_colors=num_colors)

#     mean_errors = []
#     for idx_color, op in enumerate(operators):

#         if op == "reduced-basis":
#             continue

#         mask = errors["operator"] == op

#         errors_plot = errors.loc[mask]
#         errors_plot = errors_plot.drop("operator", axis=1)
#         errors_plot = errors_plot.sort_index()

#         mean = errors_plot.mean(axis=1)
#         mean.name = op
#         mean.index.name = "p"

#         mean_errors.append(mean)

#         c = color[idx_color]
#         name = op.capitalize()

#         ax.semilogy(
#             errors_plot.index, errors_plot.iloc[:, 1:], c=c, alpha=0.5, linewidth=0.75
#         )
#         ax.semilogy(
#             errors_plot.index, errors_plot.iloc[:, 0], c=c, alpha=0.5, linewidth=0.75
#         )
#         ax.semilogy(mean.index, mean, c=c, linestyle="--", label=name, linewidth=1.5)

#     ax.grid(True)
#     ax.set_ylabel("Time avg. $L_2$ Error")
#     ax.set_ylim([1e-17, 5e0])

#     mean_errors = pd.DataFrame(mean_errors).T
#     mean_errors.to_csv(stage + ".csv")

# ax_off.tick_params(axis="y", which="both", labelleft="on", labelright="on")
# ax_on.tick_params(axis="y", which="both", labelleft="on", labelright="on")

# ax_off.set_title("Operators Error Decay (Offline)")
# ax_on.set_title("Operators Error Decay (Online)")
# ax_on.set_xlabel("Basis Percentile")
# ax_on.legend(
#     title="Avg.",
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.7),
#     ncol=3,
#     # fancybox=True, shadow=True
# )
# plt.savefig("operators_error_decay_percentile.png", **FIG_KWARGS)
# plt.close()

# -----------------------------------------------------------------------------
# Table for simulation
offline = pd.read_csv("offline.csv", index_col=0)
online = pd.read_csv("online.csv", index_col=0)
size = pd.read_csv("summary_basis.csv", index_col=0)[Treewalk.BASIS_FINAL].squeeze()
size = size.drop("reduced-basis")

online = online.sort_index(axis=1)
offline = offline.sort_index(axis=1)

offline.to_latex("errors_offline.tex", float_format="%.1e")
online.to_latex("errors_online.tex", float_format="%.1e")

basis_online = []
for operator in size.index:

    N = size.loc[operator]
    for p in online.index:
        data = (p, np.floor(N * p), operator)
        basis_online.append(data)

basis_online = pd.DataFrame(basis_online, columns=["p", "N", "operator"])
basis_online = basis_online.pivot(columns="operator", index="p")
basis_online = basis_online.astype(int)
basis_online = basis_online.sort_index(axis=1)
basis_online.to_latex("basis_online.tex")

basis_offline = []
for operator in size.index:

    N = size.loc[operator]
    for p in offline.index:
        data = (p, np.floor(N * p), operator)
        basis_offline.append(data)

basis_offline = pd.DataFrame(basis_offline, columns=["p", "N", "operator"])
basis_offline = basis_offline.pivot(columns="operator", index="p")
basis_offline = basis_offline.astype(int)
basis_offline.to_latex("basis_offline.tex")

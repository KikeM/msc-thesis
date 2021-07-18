import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from romtime.conventions import Domain
from romtime.fom import HeatEquationMovingSolver
from romtime.parameters import get_uniform_dist, round_parameters
from romtime.problems.mfp1 import define_mfp1_problem
from romtime.rom.base import Reductor
from tqdm.std import tqdm
import pandas as pd

# plt.style.use(["science", "ieee"])

domain = dict(
    L0=2.0,
    nx=500,
    nt=500,
    T=10.0,
)


# Compute omega variables
n_min = 0.4
n_max = 0.8

tf = domain["T"]
omegas = [(1.0 / tf) * np.arcsin(1.0 - n) for n in (n_min, n_max)]
omega_max = max(omegas)
omega_min = min(omegas)

grid = {
    "delta": get_uniform_dist(min=0.01, max=5.0),
    "beta": get_uniform_dist(min=0.05, max=0.1),
    "alpha_0": get_uniform_dist(min=0.01, max=2.0),
    "omega": get_uniform_dist(min=omega_min, max=omega_max),
}

_, boundary_conditions, forcing_term, u0, ue, Lt, dLt_dt = define_mfp1_problem()

# -----------------------------------------------------------------------------
# Create sampler
parameter_manager = Reductor(grid=grid)
parameter_manager.setup()
rnd = np.random.RandomState(0)
sampler = parameter_manager.build_sampling_space(num=2, rnd=rnd)


# -----------------------------------------------------------------------------
# Mesh Convergence

mu = list(sampler)[0]
mu = round_parameters(mu)

# NXs = [1e1, 1e2, 5e2, 1e3, 2e3, 5e3, 1e4]
NXs = [1e1, 5e1, 1e2, 5e2, 1e3]

index = []
_results = []
domain["nt"] = int(5e3)
for idx, nx in tqdm(enumerate(NXs), leave=True, desc="Mesh Refinement"):

    domain["nx"] = int(nx)

    fom = HeatEquationMovingSolver(
        domain=domain,
        dirichlet=boundary_conditions,
        parameters=None,
        forcing_term=forcing_term,
        u0=u0,
        filename=None,
        degrees=1,
        project_u0=False,
        exact_solution=ue,
        Lt=Lt,
        dLt_dt=dLt_dt,
    )

    fom.setup()
    fom.update_parametrization(new=mu)
    fom.solve()

    index.append(int(nx))
    tf = max(fom.errors.keys())
    final_error = fom.errors[tf]

    _results.append(final_error)

results = pd.Series(_results, index=index)
results_log = pd.Series(np.log10(_results), index=index)
results_log_log = pd.Series(np.log10(_results), index=np.log10(index))

results.to_csv("fom_mesh_convergence.csv")

# options = {"marker": "x", "linestyle": "--"}
# results.plot(**options)
# plt.grid(True)
# plt.xlabel("nx")
# plt.ylabel("Error(tf)")
# plt.show()

# results_log.plot(**options)
# plt.grid(True)
# plt.xlabel("nx")
# plt.ylabel("log10 Error(tf)")
# plt.show()

# results_log_log.plot(**options)
# plt.xlabel("log10 nx")
# plt.ylabel("log10 Error(tf)")
# plt.grid(True)
# plt.show()

# -----------------------------------------------------------------------------
# Time-Step Convergence
# mu = list(sampler)[0]
# mu = round_parameters(mu)

# # NTs = [1e1, 1e2, 5e2, 1e3, 2e3, 5e3]
# NTs = [1e1, 1e2, 2e2, 5e2, 2e3, 5e3]

# domain["nx"] = int(500)
# _results = []
# index = []
# for idx, nt in tqdm(enumerate(NTs), leave=True, desc="dt Refinement"):

#     domain["nt"] = int(nt)

#     fom = HeatEquationMovingSolver(
#         domain=domain,
#         dirichlet=boundary_conditions,
#         parameters=None,
#         forcing_term=forcing_term,
#         u0=u0,
#         filename=None,
#         degrees=1,
#         project_u0=False,
#         exact_solution=ue,
#         Lt=Lt,
#         dLt_dt=dLt_dt,
#     )

#     fom.setup()
#     fom.update_parametrization(new=mu)
#     fom.solve()

#     dt = domain["T"] / domain["nt"]
#     tf = max(fom.errors.keys())
#     final_error = fom.errors[tf]

#     index.append(dt)
#     _results.append(final_error)

# results = pd.Series(_results, index=index)
# results_log = pd.Series(np.log10(_results), index=index)
# results_log_log = pd.Series(np.log10(_results), index=np.log10(index))

# options = {"marker": "x", "linestyle": "--"}
# results.plot(**options)
# plt.grid(True)
# plt.xlabel("dt")
# plt.ylabel("Error(tf)")
# plt.show()

# results_log.plot(**options)
# plt.grid(True)
# plt.xlabel("dt")
# plt.ylabel("log10 Error(tf)")
# plt.show()

# results_log_log.plot(**options)
# plt.xlabel("log10 dt")
# plt.ylabel("log10 Error(tf)")
# plt.grid(True)
# plt.show()

# results.to_csv("fom_timestep_convergence.csv")

# -----------------------------------------------------------------------------
# Mesh velocity

# mu = list(sampler)[0]
# mu = round_parameters(mu)

# N = 5
# omegas = np.linspace(omega_min, omega_max, num=N)

# index = []
# _results = []
# for idx, omega in tqdm(enumerate(omegas), leave=True, desc="ALE velocity"):

#     fom = HeatEquationMovingSolver(
#         domain=domain,
#         dirichlet=boundary_conditions,
#         parameters=None,
#         forcing_term=forcing_term,
#         u0=u0,
#         filename=None,
#         degrees=1,
#         project_u0=False,
#         exact_solution=ue,
#         Lt=Lt,
#         dLt_dt=dLt_dt,
#     )

#     fom.setup()

#     mu["omega"] = omega
#     fom.update_parametrization(new=mu)

#     fom.solve()

#     index.append(omega)
#     tf = max(fom.errors.keys())
#     final_error = fom.errors[tf]

#     _results.append(final_error)

#     del fom

# results = pd.Series(_results, index=index)
# results_log = pd.Series(np.log10(_results), index=index)
# results_log_log = pd.Series(np.log10(_results), index=np.log10(index))

# results.to_csv("fom_ale_convergence.csv")

# options = {"marker": "x", "linestyle": "--"}
# results.plot(**options)
# plt.grid(True)
# plt.xlabel("$\omega$")
# plt.ylabel("Error(tf)")
# plt.show()

# results_log.plot(**options)
# plt.grid(True)
# plt.xlabel("$\omega$")
# plt.ylabel("log10 Error(tf)")
# plt.show()

# -----------------------------------------------------------------------------
# Plot for different parametrizations

# fom = HeatEquationMovingSolver(
#     domain=domain,
#     dirichlet=boundary_conditions,
#     parameters=None,
#     forcing_term=forcing_term,
#     u0=u0,
#     filename=None,
#     degrees=1,
#     project_u0=False,
#     exact_solution=ue,
#     Lt=Lt,
#     dLt_dt=dLt_dt,
# )

# fom.setup()

# for idx, mu in enumerate(sampler):
#     mu = round_parameters(mu)

#     fom.update_parametrization(new=mu)

#     fom.solve()

#     sol_template = f"solutions_{idx}"
#     snap_template = f"snapshots_{idx}"
#     errors_template = f"errors_{idx}"
#     fom.plot_solution(save=sol_template)
#     fom.plot_snapshots(save=snap_template)
#     fom.plot_errors(save=errors_template + "_log", log=True)
#     fom.plot_errors(save=errors_template, log=False)

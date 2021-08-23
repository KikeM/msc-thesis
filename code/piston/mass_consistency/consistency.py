"""[Gas Dynamics]
Moving Piston Problem Implementation.
"""
from functools import partial

from pprint import pprint
import fenics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from romtime.fom import OneDimensionalBurgers
from romtime.parameters import get_uniform_dist, round_parameters
from romtime.problems.piston import define_piston_problem
from romtime.utils import function_to_array, plot
from sklearn.model_selection import ParameterSampler
from romtime.rom import RomConstructorNonlinear

from romtime.conventions import BDF, Domain, OperatorType, RomParameters, Stage

fenics.set_log_level(50)

from pathlib import Path


def compute_mass_conservation_norm(mc):
    _max = np.max(mc)
    _min = np.min(mc)

    gap = (_max - _min) / _max
    gap = np.abs(gap)

    l2 = np.linalg.norm(mc)
    l2 /= len(mc)

    return l2, gap


def create_solver(L, nx, nt, tf, grid_base):
    """Solve burgers equation problem.

    Parameters
    ----------
    L : fenics.Constant
    nx : int
    nt : int
    ft : float
    parameters : tuple

    Returns
    -------
    solver : romtime.OneDimensionalHeatEquationSolver
    """

    (
        domain,
        boundary_conditions,
        forcing_term,
        u0,
        Lt,
        dLt_dt,
    ) = define_piston_problem(L, nx, tf, nt)

    solver = OneDimensionalBurgers(
        domain=domain,
        dirichlet=boundary_conditions,
        parameters=grid_base,
        forcing_term=forcing_term,
        degrees=1,
        u0=u0,
        exact_solution=None,
        Lt=Lt,
        dLt_dt=dLt_dt,
    )

    solver.setup()

    return solver


def get_parameters():

    a0 = 10.0
    omega = 26.0
    alpha = 0.000001
    delta = 0.1

    return a0, omega, alpha, delta


def get_grid_base():

    a0, omega, alpha, delta = get_parameters()
    _grid = dict(a0=a0, omega=omega, alpha=alpha, delta=delta)

    return _grid


a0, omega, alpha, delta = get_parameters()
gamma = 1.4

# Run loop
L0 = 1
nx = 1000
# nt = 1000

tf = 1.0

results = []
nts = [1e1, 1e2, 1e3, 1e4]
for idx, nt in enumerate(nts):

    nt = int(nt)

    solver = create_solver(nx=nx, nt=nt, tf=tf, L=L0, grid_base=get_grid_base())
    mu = {
        "a0": a0,
        "omega": omega,
        "alpha": alpha,
        "delta": delta,
        "gamma": gamma,
    }

    solver.BDF_SCHEME = BDF.TWO
    solver.RUNTIME_PROCESS = True

    solver.setup()

    # Update parameters
    solver.update_parametrization(new=mu)
    solver.solve()

    ts = solver.timesteps[1:]
    uh_set = list(solver.solutions.values())

    if idx == 3:
        solver.plot_probes(show=False, save="probes_hifi")

    mc, outflow, I, Iprime = solver.compute_mass_conservation(
        mu=solver.mu,
        ts=ts,
        solutions=uh_set,
        figure=False,
        title="Mass Conservation Check",
    )

    l2, gap = compute_mass_conservation_norm(mc)

    results.append([solver.dt, l2, gap])

results = pd.DataFrame(results, columns=["dt", "L2", "Gap"])
results = results.set_index("dt")
results.to_csv("time_consistency.csv")
from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from pandas.core.frame import DataFrame

from romtime.parameters import get_uniform_dist
from romtime.problems.mfp1 import HyperReducedOrderModelMoving, define_mfp1_problem
from romtime.conventions import Domain, OperatorType, RomParameters
from romtime.rom.base import Reductor
from pprint import pprint

from copy import deepcopy

from romtime.utils import bilinear_to_csr
from scipy.sparse import csr_matrix, find

A = csr_matrix(
    [
        [7.0, 8.0, 0.0],
        [-2.0, 0.0, 9.0],
        [0.0, 0.0, 9.0],
    ]
)
rows, cols, values = find(A)
A_rt = csr_matrix((values, (rows, cols)))
# csr_matrix.data

# (Pdb++) values
# array([ 7., -2.,  8.,  9.,  9.])
# (Pdb++) A.data
# array([ 7.,  8., -2.,  9.,  9.])

# (Pdb++) A.todense()
# matrix([[ 7.,  8.,  0.],
#         [-2.,  0.,  9.],
#         [ 0.,  0.,  9.]])
# (Pdb++) A_rt.todense()
# matrix([[ 7.,  8.,  0.],
#         [-2.,  0.,  9.],
#         [ 0.,  0.,  9.]])

# breakpoint()


domain = dict(
    L0=2.0,
    nx=3,
    nt=10,
    T=10.0,
)


# Compute omega variables
n_min = 0.5
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

fom_params = dict(
    domain=domain,
    dirichlet=boundary_conditions,
    forcing_term=forcing_term,
    u0=u0,
    exact_solution=ue,
    Lt=Lt,
    dLt_dt=dLt_dt,
    degrees=1,
)


reductor = Reductor(grid=grid)
rnd = np.random.RandomState(0)
mu_space = reductor.build_sampling_space(num=1, rnd=rnd)
mu_space = list(mu_space)

# -----------------------------------------------------------------------------
# ROM

EVALUATE = True
models = {
    OperatorType.MASS: True,
    OperatorType.STIFFNESS: True,
    OperatorType.CONVECTION: True,
    OperatorType.RHS: True,
}

# Reduced basis parameters
rom_params = {
    RomParameters.NUM_SNAPSHOTS: None,
    RomParameters.TOL_TIME: None,
    RomParameters.TOL_MU: None,
}

# (M)DEIM parametrization
tf, nt = domain[Domain.T], domain[Domain.NT]
ts = np.linspace(tf / nt, tf, nt)

deim_params = {
    RomParameters.TS: ts,
    RomParameters.NUM_SNAPSHOTS: None,
}
mdeim_params = {
    RomParameters.TS: ts,
    RomParameters.NUM_SNAPSHOTS: None,
    # RomParameters.NUM_MU: 2,
    # RomParameters.NUM_TIME: 50,
}

hrom = HyperReducedOrderModelMoving(
    grid=grid,
    fom_params=fom_params,
    rom_params=rom_params,
    deim_params=deim_params,
    mdeim_params=mdeim_params,
    models=models,
    rnd=rnd,
)

hrom.setup()
hrom.setup_hyperreduction()
hrom.run_offline_hyperreduction(mu_space=mu_space, evaluate=EVALUATE)

hrom.run_offline_rom(mu_space=mu_space)

_, KN_m = hrom.rom.assemble_system(mu=mu_space[0], t=0.5)
_, Kh = hrom.fom.assemble_system(mu=mu_space[0], t=0.5)

KN = hrom.rom.to_rom(Kh)

Ch = hrom.mdeim_convection.assemble(mu=mu_space[0], t=0.5)
Ch = bilinear_to_csr(Ch)
Ch_csr = Ch.copy()

Ch = Ch.todense()

Ch_int = hrom.mdeim_convection.interpolate(
    mu=mu_space[0],
    t=0.5,
    which=OperatorType.FOM,
)
Ch_int = Ch_int.todense()


Ah = hrom.mdeim_stiffness.assemble(mu=mu_space[0], t=0.5)
Ah = bilinear_to_csr(Ah)
Ch_csr = Ah.copy()

Ah = Ah.todense()

Ah_int = hrom.mdeim_stiffness.interpolate(
    mu=mu_space[0],
    t=0.5,
    which=OperatorType.FOM,
)
Ah_int = Ah_int.todense()

breakpoint()

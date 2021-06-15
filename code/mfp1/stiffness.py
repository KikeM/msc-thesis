import fenics
import numpy as np
from matplotlib import pyplot as plt
from romtime.conventions import Domain, RomParameters
from romtime.deim import MatrixDiscreteEmpiricalInterpolation
from romtime.fom import HeatEquationSolver, move_mesh
from romtime.parameters import get_uniform_dist
from romtime.problems.mfp1 import define_mfp1_problem
from romtime.rom.base import Reductor
from romtime.testing.mock import MockSolverMoving

DIFFUSION = "diffusion"
CONVECTION = "convection"
BOTH = "both"

WHICH = BOTH


class MockSolver(HeatEquationSolver):
    def __init__(
        self,
        domain: dict,
        dirichlet: dict,
        parameters: dict,
        forcing_term: str,
        u0,
        filename=None,
        poly_type="P",
        degrees=1,
        project_u0=False,
        exact_solution=None,
        Lt=None,
        dLt_dt=None,
    ) -> None:
        super().__init__(
            domain=domain,
            dirichlet=dirichlet,
            parameters=parameters,
            forcing_term=forcing_term,
            u0=u0,
            filename=filename,
            poly_type=poly_type,
            degrees=degrees,
            project_u0=project_u0,
            exact_solution=exact_solution,
            Lt=Lt,
            dLt_dt=dLt_dt,
        )

    def compute_mesh_velocity(self, mu, t):

        dLt_dt = self.dLt_dt(t=t, **mu)
        Lt = self.Lt(t=t, **mu)

        w = fenics.Expression("x[0] * dLt_dt / Lt", degree=1, dLt_dt=dLt_dt, Lt=Lt)

        return w

    def assemble_stiffness_topology(self):
        """Assemble stiffness matrix for a ALE problem.

        Parameters
        ----------
        mu : dict
        t : float
        entries : list
        """

        # ---------------------------------------------------------------------
        # Weak Formulation
        # ---------------------------------------------------------------------
        dot, dx, grad = fenics.dot, fenics.dx, fenics.grad
        u, v = self.u, self.v

        if WHICH == BOTH:
            Ah = -u.dx(0) * v * dx + dot(grad(u), grad(v)) * dx
        elif WHICH == DIFFUSION:
            Ah = dot(grad(u), grad(v)) * dx
        elif WHICH == CONVECTION:
            Ah = -u.dx(0) * v * dx

        bc = self.define_homogeneous_dirichlet_bc()
        Ah_mat = self.assemble_operator(Ah, bc)

        return Ah_mat

    @move_mesh
    def assemble_stiffness(self, mu, t, entries=None):
        """Assemble stiffness matrix for a ALE problem.

        Parameters
        ----------
        mu : dict
        t : float
        entries : [type], optional
            [description], by default None
        """

        # ---------------------------------------------------------------------
        # Weak Formulation
        # ---------------------------------------------------------------------
        dot, dx, grad = fenics.dot, fenics.dx, fenics.grad
        u, v = self.u, self.v

        w = self.compute_mesh_velocity(mu=mu, t=t)
        alpha = self.create_diffusion_coefficient(mu)

        if WHICH == BOTH:
            Ah = -w * u.dx(0) * v * dx + alpha * dot(grad(u), grad(v)) * dx
        elif WHICH == DIFFUSION:
            Ah = alpha * dot(grad(u), grad(v)) * dx
        elif WHICH == CONVECTION:
            Ah = -w * u.dx(0) * v * dx

        if entries:
            Ah_mat = self.assemble_local(form=Ah, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            Ah_mat = self.assemble_operator(Ah, bc)

        return Ah_mat


domain = dict(
    L0=2.0,
    nx=500,
    nt=250,
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

fom = MockSolver(
    domain=domain,
    dirichlet=boundary_conditions,
    parameters=None,
    forcing_term=forcing_term,
    u0=u0,
    Lt=Lt,
    dLt_dt=dLt_dt,
)

fom.setup()

# (M)DEIM parametrization
tf, nt = domain[Domain.T], domain[Domain.NT]
ts = np.linspace(tf / nt, tf, nt)

deim_params = {
    RomParameters.TS: ts,
    RomParameters.NUM_SNAPSHOTS: None,
    RomParameters.NUM_MU: 2,
    RomParameters.NUM_TIME: 2,
}

mdeim = MatrixDiscreteEmpiricalInterpolation(
    assemble=fom.assemble_stiffness,
    name="Stiffness (ALE)",
    grid=grid,
    tree_walk_params=deim_params,
)

reductor = Reductor(grid=grid)
rnd = np.random.RandomState(0)
mu_space = reductor.build_sampling_space(num=10, rnd=rnd)
mu_space = list(mu_space)

mdeim.setup(rnd=rnd)
mdeim.run(mu_space=mu_space)

mdeim.plot_spectrum(which="sigmas")
# mdeim.plot_spectrum(which="energy")

mdeim.evaluate(ts, mu_space=mu_space)
mdeim.plot_errors()

from functools import partial

import fenics
import matplotlib.pyplot as plt
import numpy as np
from romtime.base import OneDimensionalSolver
from romtime.pod import orth
from romtime.rom.base import Reductor
from romtime.utils import bilinear_to_array, function_to_array, functional_to_array
from scipy.sparse.linalg import gmres
from tqdm import tqdm


class RomConstructor(Reductor):

    FORCING = "forcing"
    LIFTING = "lifting"
    RHS = "rhs"

    GMRES_OPTIONS = dict(atol=1e-7, tol=1e-7, maxiter=1000)

    def __init__(
        self,
        fom: OneDimensionalSolver,
        grid: dict,
    ) -> None:

        super().__init__(grid=grid)

        self.fom = fom

        self.basis = None

        self.timesteps = dict()
        self.solutions = dict()
        self.liftings = dict()

        self.errors = dict()
        self.exact = dict()

        self.deim_fh = None
        self.deim_fgh = None
        self.deim_rhs = None

        self.mdeim_Mh = None
        self.mdeim_Ah = None

    def to_fom_vector(self, uN):

        V = self.basis
        uh = V.dot(uN)

        return uh

    def to_rom_vector(self, uh):

        V = self.basis
        uh_vec = function_to_array(uh)

        return V.T.dot(uh_vec)

    def to_rom_bilinear(self, Ah):

        Ah = bilinear_to_array(Ah)

        V = self.basis
        AhV = Ah.dot(V)
        AN = np.matmul(V.T, AhV)

        return AN

    def to_rom_functional(self, fh, is_array=False):

        if not is_array:
            fh = functional_to_array(fh)

        V = self.basis
        fN = V.T.dot(fh)

        return fN

    def setup(self, rnd):

        super().setup(rnd=rnd)

        self.algebraic_solver = self.create_algebraic_solver()

    def add_hyper_reductor(self, reductor, which):

        if which == self.FORCING:
            self.deim_fh = reductor
        elif which == self.LIFTING:
            self.deim_fgh = reductor
        elif which == self.RHS:
            self.deim_rhs = reductor
        else:
            raise NotImplementedError(f"Which is this reductor? {which}")

    def project_reductors(self):
        """Project collateral basis unto the solution reduced space."""

        if self.deim_fh is not None:
            self.deim_fh.project_basis(V=self.basis)
        if self.deim_fgh is not None:
            self.deim_fgh.project_basis(V=self.basis)

        if self.mdeim_Mh is not None:
            self.mdeim_Mh.project_basis(V=self.basis)
        if self.mdeim_Ah is not None:
            self.mdeim_Ah.project_basis(V=self.basis)

    def build_reduced_basis(self, num_snapshots, num_basis=None):
        """Build reduced basis.

        Parameters
        ----------
        num_snapshots : int
            Number of parameter snapshots to take.
        """

        # Create random sampler
        sampler = self.build_sampling_space(num=num_snapshots, rnd=self.random_state)

        # Put up the solver and start loop in parameter space
        fom = self.fom
        fom.setup()

        basis_time = list()
        for mu in tqdm(sampler, desc="(ROM) Building reduced basis"):

            # Save parameter
            mu_idx, mu = self.add_mu(mu=mu, step=self.OFFLINE)

            # Solve FOM time-dependent problem
            fom.update_parametrization(mu)
            fom.solve()

            # Orthonormalize the time-snapshots
            _basis, sigmas = orth(fom._snapshots)
            basis_time.append(_basis)

        basis = np.hstack(basis_time)
        self.report.update({"Basis shape after tree-walk": basis.shape})

        # Compress again all the basis
        basis, sigmas = orth(basis, num=num_basis)
        self.report.update({"Basis shape after compression": basis.shape})

        # Store reduced basis
        self.N = basis.shape[1]
        self.basis = basis

    def create_algebraic_solver(self):
        """Create algebraic solver for reduced problem

        Returns
        -------
        solver : scipy.sparse.linalg.gmres
            Iterative solver with prescribed parameters.
        """

        solver = partial(gmres, **self.GMRES_OPTIONS)

        return solver

    def solve(self, mu):
        """Solve problem with ROM.

        Parameters
        ----------
        mu : dict
            Parameter-point.
        """

        idx_mu, mu = self.add_mu(mu=mu, step=self.ONLINE)

        fom = self.fom

        # Start iteration
        solutions = dict()
        liftings = dict()

        if fom.exact_solution is not None:
            errors = dict()
            exact = dict()
        else:
            ue = None

        timesteps = [0.0]

        g, _, _ = fom.create_lifting_operator(mu=mu, t=0.0, L=fom.domain["L"])

        dt = fom.dt
        t = 0.0
        uN_n = np.zeros(shape=self.N)
        for timestep in tqdm(
            range(fom.domain["nt"]), desc="(ROM-DEIM) Online evaluation", leave=False
        ):

            # Update time
            t += dt
            g.t = t

            timesteps.append(t)

            ########################
            # Assemble linear system
            ########################
            MN_mat = self.assemble_mass(mu=mu, t=t)
            AN_mat = self.assemble_stiffness(mu=mu, t=t)
            KN_mat = MN_mat + dt * AN_mat

            fN_vec = self.assemble_rhs(mu=mu, t=t)

            bN_vec = MN_mat.dot(uN_n) + dt * fN_vec

            ###############
            # Solve problem
            ###############
            uN, info = self.algebraic_solver(A=KN_mat, b=bN_vec)

            # Update solution
            uN_n = uN.copy()

            ##############
            # FEM solution
            ##############
            gh = fenics.interpolate(g, fom.V)
            gh = function_to_array(gh)

            uh = self.to_fom_vector(uN)
            uc_h = uh + gh

            # Collect solutions
            solutions[t] = uc_h.copy()
            liftings[t] = gh.copy()

            # Compute error with exact solution
            if fom.exact_solution is not None:
                ue = fenics.Expression(fom.exact_solution, degree=2, t=t, **mu)
                ue_h = fenics.interpolate(ue, fom.V)
                ue_h = function_to_array(ue_h)
                exact[t] = ue_h.copy()

                error = self._compute_error(u=uc_h, ue=ue_h)
                errors[t] = error

        self.timesteps.update({idx_mu: timesteps})
        self.solutions.update({idx_mu: solutions})
        self.liftings.update({idx_mu: liftings})

        if ue is not None:
            self.errors.update({idx_mu: errors})
            self.exact.update({idx_mu: exact})

    def assemble_mass(self, mu, t):

        # Assemble FOM operator
        Mh = self.fom.assemble_mass(mu, t)
        MN = self.to_rom_bilinear(Mh)

        return MN

    def assemble_stiffness(self, mu, t):

        # Assemble FOM operator
        Ah = self.fom.assemble_stiffness(mu, t)
        AN = self.to_rom_bilinear(Ah)

        return AN

    def assemble_rhs(self, mu, t):
        """Assemble forcing and lifting together.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        fN : np.array
        """

        # Assemble FOM operator
        if self.deim_rhs:
            fN_vec = self.deim_rhs.interpolate(mu=mu, t=t, which=self.deim_rhs.ROM)
        else:
            fh = self.fom.assemble_forcing(mu, t)
            fgh = self.fom.assemble_lifting(mu, t)

            fN = self.to_rom_functional(fh, is_array=False)
            fgN = self.to_rom_functional(fgh, is_array=False)
            fN_vec = fN + fgN

        return fN_vec

    def assemble_forcing(self, mu, t):
        """Assemble reduced forcing term.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        fN : np.array
        """

        # Assemble FOM operator
        if self.deim_fh:
            fN = self.deim_fh.interpolate(mu=mu, t=t, which=self.deim_fh.ROM)
        else:
            fh = self.fom.assemble_forcing(mu, t)
            fN = self.to_rom_functional(fh, is_array=False)

        return fN

    def assemble_lifting(self, mu, t):
        """Assemble reduced lifting term.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        fgN : np.array
        """

        if self.deim_fgh:
            fgN = self.deim_fgh.interpolate(mu=mu, t=t, which=self.deim_fh.ROM)
        else:
            fgh = self.fom.assemble_lifting(mu, t)
            fgN = self.to_rom_functional(fgh, is_array=False)

        return fgN
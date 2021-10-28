# msc-thesis

M. Sc. Thesis code and results.

The computation engine is located at https://github.com/KikeM/romtime.

[Defense and Graduation tasks](https://github.com/KikeM/msc-thesis/issues?q=is%3Aissue+is%3Aopen+label%3Agreen-light-meeting)

## Abstract

We present a Reduced Order Model (ROM) for a one-dimensional gas dynamics problem:
   the isentropic piston.
   The main body of the PDE, 
   the geometrical definition of the moving boundary, 
   and the boundary conditions themselves are parametrized.
   The Full Order Model is obtained with a Galerkin semi-implicit Finite Element discretization,
   under the Arbitrary-Lagrangian formulation (ALE).
   To stabilize the system, an artificial viscosity term is included.
   The Reduced Basis to express the solution is obtained with the classical POD technique.
   
   To overcome the explicit use of the jacobian transformation, 
   typical in the context of moving domains,
   a system approximation technique is used.
   The (Matrix) Discrete Empirical Interpolation Method, (M)DEIM, allows us
   to work with a weak form defined in the physical domain (and hence the physical weak formulation)
   whilst maintaining an
   efficient assembly for the algebraic operators, 
   despite their evolution with every timestep.
   
   All in all, our approach to the construction of the Reduced Order Model is purely algebraic
   and makes no use of Full Order structures in its resolution, 
   thus achieving a perfect \textit{offline-online} split.
   A concise description of the reducing procedure is provided, 
   together with a posteriori error estimations, obtained via model truncation,
   to certify the Reduced Order Model.

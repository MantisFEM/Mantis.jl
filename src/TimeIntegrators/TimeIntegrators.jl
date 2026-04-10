module TimeIntegrators

using LinearAlgebra
using StaticArrays
import SparseArrays

# These exports make these symbols/functions/etc. available outside the Forms module. Only
# if they are also exported in the Mantis.jl file will they be available when using Mantis.

# Abstract types
# The public keyword is only available in Julia 1.11 and up. Since we also support LTS
# (currently 1.10), we add the following line from the manual:
# https://docs.julialang.org/en/v1.12/manual/modules/#Export-lists
VERSION >= v"1.11.0-DEV.469" && eval(
    Meta.parse(
        "public AbstractTimeIntegrator",
    ),
)

# Type parameter methods
export get_num_stages, get_num_steps
# Problem-specific information
export TimeIntegrationOperators, define_explicit_ode, define_diagonally_implicit_ode,
    define_implicit_ode, define_imex_ode
# TimeLevels
export TimeLevels
# Integrator (Types)
export check_implicit, Explicit, DiagonallyImplicit, Implicit, IMEX, get_order
# Solution objects
export TimeIntegrationSolution, get_num_variables, get_solution, get_scheme,
    get_startup_scheme, get_remaining_startup_steps

# Schemes
export butcher_tableau_to_glm, FORWARD_EULER, EXPLICIT_MIDPOINT, HEUN2, RALSTON2, HEUN3,
    RK3, RALSTON3, VDHW3, SSPRK3, RK4, RK4_3_8, RALSTON4, BACKWARD_EULER, RADAU_IA_1,
    IMPLICIT_MIDPOINT, DIRK2, DIRK3, RADAU_IA_3, DIRK4, GAUSS_LEGENDRE_4, GAUSS_LEGENDRE_6,
    AB1, AB2, AB3, AB4, BDF1, BDF2, BDF3, BDF4, AM0, AM1, AM2, AM3, AM4,
    BACKWARD_FORWARD_EULER, MIDPOINT_IMEX, RK3_IMEX, CNAB2, SSSS2

# Initialisations
export initialize_scheme

# Integrations
export time_integrate, time_integrate!


include("Definitions.jl")
include("Schemes.jl")
include("Initialisations.jl")
include("Integrations.jl")

end

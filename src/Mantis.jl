module Mantis

############################################################################################
#                                         Includes                                         #
############################################################################################
include("GeneralHelpers/GeneralHelpers.jl")
include("Mesh/Mesh.jl")
include("Points/Points.jl")
include("Hierarchy/Hierarchy.jl")
include("Geometry/Geometry.jl")
include("FunctionSpaces/FunctionSpaces.jl")
include("Quadrature/Quadrature.jl")
include("Forms/Forms.jl")
include("Analysis/Analysis.jl")
include("Assemblers/Assemblers.jl")
include("TimeIntegrators/TimeIntegrators.jl")
include("Plot/Plot.jl")

############################################################################################
#                                         Exports                                          #
############################################################################################
# Exported modules. Note that GeneralHelpers is not explicitly exported.
export Mesh,
    Points,
    Quadrature,
    Hierarchy,
    FunctionSpaces,
    Geometry,
    Forms,
    Analysis,
    Assemblers,
    TimeIntegrators,
    Plot

# Exports from Forms.
using .Forms
export d, ★, ♯, ∧, ∫, dstar, δ
export ConstantFormSpace, FormField, AnalyticalFormField, FormSpace
export evaluate
export get_label, get_num_basis, get_coefficients

# Exports from TimeIntegrators
using .TimeIntegrators
export define_explicit_ode,
    define_diagonally_implicit_ode, define_implicit_ode, define_imex_ode
export get_solution, initialise_scheme, time_integrate, time_integrate!

end

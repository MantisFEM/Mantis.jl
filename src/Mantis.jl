module Mantis

############################################################################################
#                                         Includes                                         #
############################################################################################
include("GeneralHelpers/GeneralHelpers.jl")  # Creates Module GeneralHelpers
include("Mesh/Mesh.jl")  # Creates Module Mesh
include("Points/Points.jl")  # Creates Module Mesh
include("Hierarchy/Hierarchy.jl") # Creates Module Hierarchy
include("Geometry/Geometry.jl")  # Creates Module Geometry
include("FunctionSpaces/FunctionSpaces.jl")  # Creates Module FunctionSpaces
include("Quadrature/Quadrature.jl")  # Creates Module Quadrature
include("Forms/Forms.jl")  # Creates Module Forms
include("Analysis/Analysis.jl")  # Creates Module Analysis
include("Assemblers/Assemblers.jl")  # Creates Module Assemblers
include("TimeIntegrators/TimeIntegrators.jl")  # Creates Module TimeIntegrators
include("Plot/Plot.jl")  # Creates Module Plot

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

end

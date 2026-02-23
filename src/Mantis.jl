module Mantis

############################################################################################
#                                         Includes                                         #
############################################################################################
include("GeneralHelpers/GeneralHelpers.jl")  # Creates Module GeneralHelpers
include("Topology/Topology.jl")  # Creates Module Topology
include("Points/Points.jl")  # Creates Module Points
include("Geometry/Geometry.jl")  # Creates Module Geometry
include("FunctionSpaces/FunctionSpaces.jl")  # Creates Module FunctionSpaces
include("Quadrature/Quadrature.jl")  # Creates Module Quadrature
include("Forms/Forms.jl")  # Creates Module Forms
include("Analysis/Analysis.jl")  # Creates Module Analysis
include("Assemblers/Assemblers.jl")  # Creates Module Assemblers
include("Plot/Plot.jl")  # Creates Module Plot

############################################################################################
#                                         Exports                                          #
############################################################################################
export Topology,
    Points, Quadrature, FunctionSpaces, Geometry, Forms, Analysis, Assemblers, Plot
include("../exports/Exports.jl")

end

module Mantis

# Remember that these include statements have to be in order, so that 
# the include that are listed later can use code from the previous ones, 
# but not the other way around.
include("Mesh/mesh.jl")  # Creates Module Mesh
include("Polynomials/Polynomials.jl")  # Creates Module Polynomials
include("Quadrature/Quadrature.jl")  # Creates Module Quadrature
include("FunctionSpaces/FunctionSpaces.jl")  # Creates Module FunctionSpaces
include("Hierarchy/Hierarchy.jl")  # Creates Module Hierarchy

end

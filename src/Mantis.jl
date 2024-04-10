module Mantis

# Remember that these include statements have to be in order, so that 
# the include that are listed later can use code from the previous ones, 
# but not the other way around.
include("Mesh/mesh.jl")  # Creates Module Mesh
include("ElementSpaces/ElementSpaces.jl")  # Creates Module ElementSpaces
include("Quadrature/Quadrature.jl")  # Creates Module Quadrature
include("FiniteElementSpaces/FiniteElementSpaces.jl")  # Creates Module FiniteElementSpaces
include("HierarchicalFiniteElementSpaces/HierarchicalFiniteElementSpaces.jl")  # Creates Module HierarchicalFiniteElementSpaces

end

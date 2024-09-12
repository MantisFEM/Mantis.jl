include("HelperFunctions.jl")
# AbstractFiniteElementSpace: Common functionality accross FEM spaces
include("AbstractFiniteElementSpace.jl")
# Extraction operator: functionaly for extraction operator
include("ExtractionOperator.jl")
# rational version of finite element spaces
include("RationalFiniteElementSpaces.jl")
# canonical finite element space wrapper
include("CanonicalFiniteElementSpaces.jl")
# univariate function spaces
include("UnivariateSplineSpaces.jl")
include("UnivariateSplineExtractions.jl")
# composite function spaces
include("UnstructuredSpaces.jl")
include("TensorProductSpaces.jl")
# two scale relations
include("TwoScaleRelations/TwoScaleRelations.jl")
# hierarchical function spaces
include("HierarchicalFiniteElementSpaces.jl")
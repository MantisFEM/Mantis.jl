"""
	module Hierarchical

Contains all methods related with hierarchies that are object-agnostic. In essence, the
methods in this module provide a base, general logic for hierarchical refinements — be those
hierarchical geometries or function spaces, for example.
"""
module Hierarchical

using ..TensorProducts

import SparseArrays
import LinearAlgebra

############################################################################################
#                                         Exports                                          #
############################################################################################

# Refinement
## Structures
export AbstractRefinement, Refinement, RefinementExplicit, RefinementMethod

# Scalings
## Structures
export AbstractScaling,
    MatrixScaling,
    RelationEmpty,
    RelationExplicit,
    RelationMethod,
    RelationType,
    Relations,
    Scaling
## Methods
export get_child, get_children, get_parent, get_parents, get_sets

# ActiveInfo
## Structures
export ActiveInfo
## Methods
export add_level!,
    convert_to_hier_id,
    convert_to_level_and_level_id,
    convert_to_level_id,
    get_level,
    get_level_ids,
    get_level_set,
    get_level_sets,
    get_num_levels,
    get_num_objects,
    update!

# Hierarchy
## Structures
export AbstractHierarchy, Hierarchy, NestedHierarchy
## Methods
export get_active_info, get_descendants, get_nested_ids, get_scaling, get_scalings

############################################################################################
#                                         Includes                                         #
############################################################################################

# How to produce new sets
include("Refinement.jl")
# How parents/children are related
include("Scalings.jl")
# Which parents/children are active
include("ActiveInfo.jl")
# Combination of the previous
include("Hierarchy.jl")

end

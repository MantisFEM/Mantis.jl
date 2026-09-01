"""
	module Hierarchical

Utilities related to hierarchies of abstract objects. Includes methods one refinement,
scalings between parent and child sets, and information about active objects at each level.
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
export convert_to_hier_id,
    convert_to_level_and_level_id,
    convert_to_level_id,
    get_level,
    get_level_ids,
    get_level_set,
    get_level_sets,
    get_num_levels,
    get_num_objects,
    refine,
    update

# Hierarchy
## Structures
export AbstractHierarchy, Hierarchy, TreeHierarchy
## Methods
export get_active_info, get_descendants, get_tree_ids, get_scaling, get_scalings

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

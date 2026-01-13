# """
# This (sub-)module provides an evaluation mask. An evaluation mask describes how to map
# from a set of evaluation points on a manifold to a set of evaluation points on a base mesh.
# This is useful for example when evaluating a function on a finer mesh than the one used
# for integration.
# """

"""
    AbstractEvaluationMask{manifold_dim}

Supertype for all evaluation maskes. An evaluation mask describes how to map from a set of
evaluation points on a manifold to a set of evaluation points on a base mesh. This is useful
for example when evaluating a function on a finer mesh than the one used for integration.
"""
abstract type AbstractEvaluationMask{manifold_dim} end

"""
    get_manifold_dim(eval_mask::AbstractEvaluationMask)

Get the dimension of the manifold.

# Arguments
- `eval_mask::AbstractEvaluationMask`: The evaluation mask.

# Returns
- `manifold_dim::Int`: The dimension of the manifold.
"""
function get_manifold_dim(::AbstractEvaluationMask{manifold_dim}) where {manifold_dim}
    return manifold_dim
end

"""
    get_num_elements(eval_mask::AbstractEvaluationMask)

Get the number of elements in the evaluation mask.

# Arguments
- `eval_mask::AbstractEvaluationMask`: The evaluation mask.

# Returns
- `num_elements::Int`: The number of elements.
"""
function get_num_elements(eval_mask::AbstractEvaluationMask)
    return eval_mask.num_elements
end

"""
    get_num_elements_base(eval_mask::AbstractEvaluationMask)

Get the number of elements in the base mesh.

# Arguments
- `eval_mask::AbstractEvaluationMask`: The evaluation mask.

# Returns
- `num_elements_base::Int`: The number of elements in the base mesh.
"""
function get_num_elements_base(eval_mask::AbstractEvaluationMask)
    return eval_mask.num_elements_base
end

"""
    transform_evaluation_points(
        eval_mask::AffineEvaluationMask{manifold_dim, num_elements, num_elements_base},
        element_id::Int,
		xi::Points.AbstractPoints{manifold_dim}
    )

Evaluate the mapping from the evaluation mask to the base.

# Arguments
- `eval_mask::AffineEvaluationMask`: The evaluation mask.
- `element_id::Int`: The evaluation element index.
- `xi::Points.AbstractPoints{manifold_dim}`: The canonical points for the evaluation mask.

# Returns
- `element_id_base::Int`: The element index in the range geometry.
- `xi_base::NTuple{manifold_dim, T}`: The canonical points for the base.
- `scaling::NTuple{manifold_dim, Float64}`: The scaling of the element.
"""
function transform_evaluation_points(
    eval_mask::AbstractEvaluationMask{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    return throw(MethodError(transform_evaluation_points, (eval_mask, element_id, xi)))
end

include("./AffineEvaluationMask.jl")
include("./EvaluationMaskHelpers.jl")

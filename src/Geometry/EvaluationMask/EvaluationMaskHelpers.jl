################################################################################
# Hierarchical and/or trimmed evaluation maskes
################################################################################

"""
    subdivide_evaluation_mask(eval_mask, nsubd; elements_to_exclude)

Create a hierarchical evaluation mask by subdividing the `i`-th element of the evaluation
mesh into `nsubd[i][j]` elements in the `j`-th direction. The `elements_to_exclude` argument
is a vector of element indices to exclude from the mesh.

# Arguments
- `eval_mask::AbstractEvaluationMask`: The evaluation mask.
- `nsubd::Vector{NTuple{manifold_dim, Int}}`: The number of subdivisions in each direction.
- `elements_to_exclude::AbstractVector{Int}`: The elements to exclude.

# Returns
- `subdiv_eval_mask::AbstractEvaluationMask`: The subdivided evaluation mask.
"""
function subdivide_evaluation_mask(
    eval_mask::AbstractEvaluationMask{manifold_dim, num_elements, num_elements_base},
    nsubd::Vector{NTuple{manifold_dim, Int}};
    elements_to_exclude::AbstractVector{Int}=Int[],
) where {manifold_dim, num_elements, num_elements_base}

    # number of new elements
    num_new_elements = sum(map(prod, nsubd))

    # allocate memory for evaluation mask
    element_idx_map = zeros(Int, num_new_elements)
    translations = Vector{NTuple{manifold_dim, Float64}}(undef, num_new_elements)
    scalings = Vector{NTuple{manifold_dim, Float64}}(undef, num_new_elements)

    # build all the maps
    count = 0
    for i in 1:num_elements
        k = [1:nsubd[i][j] for j in 1:manifold_dim]
        for idx in Iterators.product(k...)
            count += 1
            element_idx_map[count] = i
            translations[count] = (idx .- 1) ./ nsubd[i]
            scalings[count] = 1 ./ nsubd[i]
        end
    end

    # build the new evaluation mask relative to the old one
    new_eval_mask = AffineEvaluationMask(
        num_new_elements, num_elements, element_idx_map, translations, scalings
    )

    # trim the new evaluation mask
    if length(elements_to_exclude) > 0
        new_eval_mask = trim_evaluation_mask(new_eval_mask, elements_to_exclude)
    end

    return compose_evaluation_masks(new_eval_mask, eval_mask)
end

"""
    subdivide_evaluation_mask(eval_mask, nsubd; elements_to_exclude)

Create a hierarchical evaluation mask by subdividing the `i`-th element of the evaluation
mesh into `nsubd[i][j]` elements in the `j`-th direction. The `elements_to_exclude`
argument is a vector of element indices to exclude from the mesh.

# Arguments
- `eval_mask::AbstractEvaluationMask`: The evaluation mask.
- `nsubd::Vector{Int}`: The number of subdivisions for each element.
- `elements_to_exclude::Vector{Int}`: The elements to exclude.

# Returns
- `subdiv_eval_mask::AbstractEvaluationMask`: The subdivided evaluation mask.
"""
function subdivide_evaluation_mask(
    eval_mask::AbstractEvaluationMask{manifold_dim, num_elements, num_elements_base},
    nsubd::Vector{Int};
    elements_to_exclude::AbstractVector{Int}=Int[],
) where {manifold_dim, num_elements, num_elements_base}
    tones = ntuple(_ -> 1, manifold_dim)
    nsubd = [tones .* nsubd[i] for i in 1:num_elements]

    return subdivide_evaluation_mask(
        eval_mask, nsubd; elements_to_exclude=elements_to_exclude
    )
end

"""
    subdivide_evaluation_mask(eval_mask, nsubd; elements_to_exclude)

Create a hierarchical evaluation mask by subdividing the `i`-th element of the evaluation
mesh into `nsubd[i][j]` elements in the `j`-th direction. The `elements_to_exclude`
argument is a vector of element indices to exclude from the mesh.

# Arguments
- `eval_mask::AbstractEvaluationMask`: The evaluation mask.
- `nsubd::Int`: The number of subdivisions for each element.
- `elements_to_exclude::AbstractVector{Int}`: The elements to exclude.

# Returns
- `subdiv_eval_mask::AbstractEvaluationMask`: The subdivided evaluation mask.
"""
function subdivide_evaluation_mask(
    eval_mask::AbstractEvaluationMask{manifold_dim, num_elements, num_elements_base},
    nsubd::Int;
    elements_to_exclude::AbstractVector{Int}=Int[],
) where {manifold_dim, num_elements, num_elements_base}
    return subdivide_evaluation_mask(
        eval_mask, nsubd .* ones(Int, num_elements); elements_to_exclude=elements_to_exclude
    )
end

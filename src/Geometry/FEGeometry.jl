
"""
    FEGeometry{manifold_dim, image_dim, num_patches, F} <: AbstractGeometry{manifold_dim, image_dim, num_patches}

Geometry defined from a finite element space `fe_space` and a matrix of geometric
coefficients `geometry_coeffs`.

# Fields
- `geometry_coeffs::Matrix{Float64}`: The coefficients used to linearly combine the basis
    functions in `fe_space` to generate the geometry. The size of `geometry_coeffs` is
    `(num_basis, image_dim)`, where `num_basis` corresponds to the number of basis functions
    that span `fe_space` and `image_dim` the number of dimensions in the resulting
    geometry.
- `fe_space::F`: Finite element space used to define the geometry.
- `num_elements::Int`: The number of elements in the geometry, given by the number of
    elements in the finite element space.

# Type parameters
- `manifold_dim`: Dimension of the domain in `fe_space`.
- `F <: FunctionSpaces.AbstractFESpace{manifold_dim, image_dim, num_patches}`: Underlying finite element space of
    the geometry.

# Inner Constructors
- `FEGeometry(fe_space::F, geometry_coeffs::Matrix{Float64})`: Constructs the FEGeometry
    from a finite element space `fe_space` and a set of pre-defined `geometry_coeffs`,
    deducing the number of elements from `fe_space`.

# Outer Constructors
- [`compute_parametric_geometry`](@ref).
"""
struct FEGeometry{manifold_dim, image_dim, num_patches, F} <: AbstractGeometry{manifold_dim, image_dim, num_patches}
    geometry_coeffs::Matrix{Float64}
    fe_space::F
    num_elements::Int

    function FEGeometry(
        fe_space::F, geometry_coeffs::Matrix{Float64}
    ) where {manifold_dim, num_patches, F <: FunctionSpaces.AbstractFESpace{manifold_dim, num_patches}}
        num_elements = FunctionSpaces.get_num_elements(fe_space)
        image_dim = size(geometry_coeffs, 2)

        return new{manifold_dim, image_dim, num_patches, F}(geometry_coeffs, fe_space, num_elements)
    end
end

function get_fe_space(geometry::FEGeometry)
    return geometry.fe_space
end

"""
    compute_parametric_geometry(fe_space::FunctionSpaces.AbstractFESpace)

Returns the parametric geometry associated with `fe_space` by computing the geometry
coefficients of the space.

# Arguments
- `fe_space::FunctionSpaces.AbstractFESpace`: Finite element space
    for which to compute the geometry.

# Returns
- `::FEGeometry`: structure of the finite element geometry.
"""
function compute_parametric_geometry(fe_space::FunctionSpaces.AbstractFESpace)
    geometry_coefficients = FunctionSpaces._compute_parametric_geometry_coeffs(fe_space)

    return FEGeometry(fe_space, geometry_coefficients)
end
# These functions will need to be updated, since they currently assume a single patch
# geometry. This will also change when the dependency on FunctionSpaces is flipped.
function get_parametric_geometry(geometry::FEGeometry, patch_id::Int=1)
    @warn "The `patch_id` argument is currently ignored."
    return compute_parametric_geometry(get_fe_space(geometry))
end

function get_element_measure(geometry::FEGeometry, element_id::Int)
    return FunctionSpaces.get_element_measure(get_fe_space(geometry), element_id)
end

function get_element_lengths(geometry::FEGeometry, element_id::Int)
    return FunctionSpaces.get_element_lengths(get_fe_space(geometry), element_id)
end

function evaluate(
    geometry::FEGeometry{manifold_dim, image_dim, num_patches, F},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, F}
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(
        get_fe_space(geometry), element_id, xi, 0
    )
    length
    num_eval_points = Points.get_num_points(xi)
    eval = zeros(Float64, num_eval_points, image_dim)

    for dim in 1:image_dim
        for cartesian_id in CartesianIndices(fem_basis[1][1][1])
            (point, basis_id) = Tuple(cartesian_id)
            eval[point, dim] +=
                fem_basis[1][1][1][point, basis_id] *
                geometry.geometry_coeffs[fem_basis_indices[basis_id], dim]
        end
    end

    return eval
end

function jacobian(
    geometry::FEGeometry{manifold_dim, image_dim, num_patches, F},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, F}
    # Jᵢⱼ = ∂Φⁱ\∂ξⱼ
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(
        get_fe_space(geometry), element_id, xi, 1
    )

    # Generate derivatives indices. For derivative order 1, each dimension is derivated
    # once. Then, the corresponding derivative index for the given key is computed.
    der_idxs = ntuple(manifold_dim) do k
        key = zeros(Int, manifold_dim)
        key[k] = 1
        der_idx = FunctionSpaces.get_derivative_idx(key)

        return der_idx
    end

    num_eval_points = Points.get_num_points(xi)
    J = zeros(num_eval_points, image_dim, manifold_dim)
    for basis_id in eachindex(fem_basis_indices)
        for cartesian_idx in CartesianIndices(J)
            (point, k_im, k_mani) = Tuple(cartesian_idx)
            J[point, k_im, k_mani] +=
                fem_basis[2][der_idxs[k_mani]][1][point, basis_id] *
                geometry.geometry_coeffs[fem_basis_indices[basis_id], k_im]
        end
    end

    return J
end

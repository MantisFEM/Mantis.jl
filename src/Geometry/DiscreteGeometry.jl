############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    DiscreteGeometry{manifold_dim, image_dim, num_patches, F} <: AbstractGeometry{
        manifold_dim, image_dim, num_patches
    }

A geometry defined in terms of an `evaluable_function` over a given number of elements and
manifold dimension.

# Fields
- `evaluable_function::F`: Function that, given an `element_id` and local coordinates `xi`,
    returns values and derivatives in the same nested format as a finite element space
    (see [FunctionSpaces](@ref)). In particular, the output structure must be compatible
    with `FunctionSpaces.AbstractFESpace` so that geometry evaluations and Jacobians can be
    computed element-wise.
- `num_elements::Int`: Total number of elements in the discrete geometry.

# Type parameters
- `manifold_dim`: Dimension of the reference (parametric) domain.
- `image_dim`: Dimension of the physical embedding space.
- `num_patches`: Number of patches of the geometry.
- `F`: The type of the evaluable function.
"""
struct DiscreteGeometry{manifold_dim, image_dim, num_patches, F, L, NP} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    evaluable_function::F
    element_length_function::L
    num_elements::Int
    num_elements_per_patch::NP

    function DiscreteGeometry(
        manifold_dim::Int,
        image_dim::Int,
        num_elements::Int,
        evaluable_function::F,
        element_length_function::L,
        num_elements_per_patch::NP,
    ) where {num_patches, F <: Function, L <: Function, NP <: NTuple{num_patches, Int}}
        return new{manifold_dim, image_dim, num_patches, F, L, NP}(
            evaluable_function,
            element_length_function,
            num_elements,
            num_elements_per_patch,
        )
    end
end

function DiscreteGeometry(
    manifold_dim::Int,
    num_elements::Int,
    evaluable_function::Function,
    element_length_function::Function,
)
    return DiscreteGeometry(
        manifold_dim,
        manifold_dim,
        num_elements,
        (num_elements,),
        evaluable_function,
        element_length_function,
    )
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_evaluable_function(geometry::DiscreteGeometry) = geometry.evaluable_function
get_element_length_function(geometry::DiscreteGeometry) = geometry.element_length_function

function get_element_lengths(geometry::DiscreteGeometry, element_id::Int)
    return get_element_length_function(geometry)(element_id)
end

############################################################################################
#                                        Evaluation                                        #
############################################################################################

function evaluate(
    geometry::DiscreteGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    return get_evaluable_function(geometry)(element_id, xi)[1][1]
end

function jacobian(
    geometry::DiscreteGeometry{manifold_dim, image_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim}
    # evaluate first derivatives of geometry
    x = get_evaluable_function(geometry)(element_id, xi, 1)
    # Generate derivatives indices. For derivative order 1, each dimension is derivated
    # once. Then, the corresponding derivative index for the given key is computed.
    der_idxs = ntuple(manifold_dim) do k
        key = ntuple(manifold_dim) do dim
            return dim == k ? 1 : 0
        end
        der_idx = GeneralHelpers.get_derivative_idx(key)

        return der_idx
    end

    # Compute Jacobian and return
    num_eval_points = Points.get_num_points(xi)
    J = Vector{SMatrix{image_dim, manifold_dim, Float64, image_dim * manifold_dim}}(
        undef, num_eval_points
    )
    cartesian_idxs = CartesianIndices((image_dim, manifold_dim))
    for point in eachindex(J)
        Jp = zeros(image_dim, manifold_dim)
        for cartesian_idx in cartesian_idxs
            (k_im, k_mani) = Tuple(cartesian_idx)
            Jp[k_im, k_mani] = x[2][der_idxs[k_mani]][point, k_im]
        end

        J[point] = SMatrix{image_dim, manifold_dim}(Jp)
    end

    return J
end

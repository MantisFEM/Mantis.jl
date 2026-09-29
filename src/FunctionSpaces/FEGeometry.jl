############################################################################################
#                                        FEGeometry                                        #
############################################################################################

"""
    FEGeometry{manifold_dim, image_dim, num_patches, T, F, V} <:
    Geometry.AbstractGeometry{manifold_dim, image_dim, num_patches}

Geometry represented using an [`AbstractFESpace`](@ref).

!!! note "FEGeometry is defined in FunctionSpaces."
    The FEGeometry struct is defined inside the FunctionSpaces module. So, when creating an
    FEGeometry, you should use FunctionSpaces.FEGeometry. Alternatively, if you are
    `using Mantis`, you can call FEGeometry directly (it is exported).
"""
struct FEGeometry{manifold_dim, image_dim, num_patches, T, F, V} <:
       Geometry.AbstractGeometry{manifold_dim, image_dim, num_patches}
    topology::T
    space::F
    coefficients::Matrix{V}

    function FEGeometry(
        space::F,
        coefficients::Matrix{V},
        topology::T=Geometry.get_topology(get_geometry(space)),
    ) where {
        manifold_dim,
        ir_dim,
        num_patches,
        F <: AbstractFESpace{manifold_dim, 1, num_patches},
        T <: Topology.MeshTopology{manifold_dim, ir_dim, num_patches},
        V <: Real,
    }
        return new{manifold_dim, size(coefficients, 2), num_patches, T, F, V}(
            topology, space, coefficients
        )
    end
end

function get_space(geometry::FEGeometry)
    return geometry.space
end
function get_coefficients(geometry::FEGeometry)
    return geometry.coefficients
end

function Geometry.get_num_elements(geometry::FEGeometry)
    return Geometry.get_num_elements(get_geometry(get_space(geometry)))
end

function Geometry.get_num_elements_per_patch(geometry::FEGeometry)
    return Geometry.get_num_elements_per_patch(get_geometry(get_space(geometry)))
end

function Geometry.get_elements(
    geometry::FEGeometry,
    patch_id,
    local_object_id,
    geometric_dim,
    rotation=0,
    orientation=1,
)
    return Geometry.get_elements(
        get_geometry(get_space(geometry)),
        patch_id,
        local_object_id,
        geometric_dim,
        rotation,
        orientation,
    )
end

function Geometry.get_element_vertices(geometry::FEGeometry, element_id::Int)
    return Geometry.get_element_vertices(get_geometry(get_space(geometry)), element_id)
end

function Geometry.get_element_lengths(geometry::FEGeometry, element_id::Int)
    return Geometry.get_element_lengths(get_geometry(get_space(geometry)), element_id)
end

function Geometry.get_element_measure(geometry::FEGeometry, element_id::Int)
    return Geometry.get_element_measure(get_geometry(get_space(geometry)), element_id)
end

function Geometry.evaluate(
    geometry::FEGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    space = get_space(geometry)
    coefficients = get_coefficients(geometry)

    return mapreduce(
        c -> evaluate(space, element_id, xi, 0, c)[1][1][1], hcat, eachcol(coefficients)
    )
end

function Geometry.jacobian(
    geometry::FEGeometry{manifold_dim, image_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim}
    space = get_space(geometry)
    coefficients = get_coefficients(geometry)
    x_per_image_dim = ntuple(image_dim) do c
        return evaluate(space, element_id, xi, 1, coefficients[:, c])
    end
    T = eltype(eltype(eltype(eltype(eltype(x_per_image_dim)))))

    # Generate derivatives indices. For derivative order 1, each dimension is derivated
    # once. Then, the corresponding derivative index for the given key is computed.
    der_idxs = ntuple(manifold_dim) do k
        key = ntuple(manifold_dim) do dim
            return dim == k ? 1 : 0
        end

        return GeneralHelpers.get_derivative_idx(key)
    end

    num_eval_points = Points.get_num_points(xi)
    J = Vector{StaticArrays.SMatrix{image_dim, manifold_dim, T, image_dim * manifold_dim}}(
        undef, num_eval_points
    )
    for point in eachindex(J)
        J[point] = StaticArrays.SMatrix{image_dim, manifold_dim, T}(
            x_per_image_dim[ki][2][der_idxs[km]][1][point] for
            km in 1:manifold_dim, ki in 1:image_dim
        )
    end

    return J
end

function get_der_idx_from(i, j, ::Val{manifold_dim}) where {manifold_dim}
    if i == j
        key = ntuple(dim -> dim == i ? 2 : 0, manifold_dim)
        return GeneralHelpers.get_derivative_idx(key)
    else
        key = ntuple(dim -> dim == i ? 1 : (dim == j ? 1 : 0), manifold_dim)
        return GeneralHelpers.get_derivative_idx(key)
    end
end

function Geometry.hessian(
    geometry::FEGeometry{manifold_dim, image_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim}
    space = get_space(geometry)
    coefficients = get_coefficients(geometry)
    x_per_image_dim = ntuple(image_dim) do c
        return evaluate(space, element_id, xi, 2, coefficients[:, c])
    end
    T = eltype(eltype(eltype(eltype(eltype(x_per_image_dim)))))

    num_eval_points = Points.get_num_points(xi)

    return [
        ntuple(image_dim) do ki
            return StaticArrays.SMatrix{manifold_dim, manifold_dim, T}(
                x_per_image_dim[ki][3][get_der_idx_from(km1, km2, Val(manifold_dim))][1][p]
                for km1 in 1:manifold_dim, km2 in 1:manifold_dim
            )
        end

        for p in 1:num_eval_points
    ]
end

# function DiscreteGeometry(
#     space::S, coefficients::Matrix{T}
# ) where {
#     manifold_dim, num_patches, S <: AbstractFESpace{manifold_dim, 1, num_patches}, T <: Real
# }
#     image_dim = size(coefficients, 2)
#     num_elements = get_num_elements(space)
#     num_elements_per_patch = get_num_elements_per_patch(space)
#     function evaluable_function(
#         element_id::Int, xi::Points.AbstractPoints{manifold_dim}, num_derivatives::Int=0
#     )
#         space_basis, basis_indices = evaluate(space, element_id, xi, num_derivatives)
#         eval = Vector{Vector{Matrix{Float64}}}(undef, num_derivatives + 1) # 1 component
#         num_points = Points.get_num_points(xi)
#         for i in eachindex(eval)
#             num_ders = length(space_basis[i])
#             eval[i] = Vector{Matrix{Float64}}(undef, num_ders)
#             for j in eachindex(eval[i])
#                 eval[i][j] = Matrix{Float64}(undef, num_points, image_dim)
#                 for l in 1:image_dim
#                     eval[i][j][:, l] = space_basis[i][j][1] * coefficients[basis_indices, l]
#                 end
#             end
#         end

#         return eval
#     end

#     function element_length_function(element_id::Int)
#         return Geometry.get_element_lengths(get_geometry(space), element_id)
#     end

#     return Geometry.DiscreteGeometry(
#         manifold_dim,
#         image_dim,
#         num_elements,
#         evaluable_function,
#         element_length_function,
#         num_elements_per_patch,
#     )
# end

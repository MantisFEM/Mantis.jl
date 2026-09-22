############################################################################################
#                                     DiscreteGeometry                                     #
############################################################################################

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

function Geometry.get_elements(
    geometry::FEGeometry, patch_id, local_object_id, geometric_dim
)
    return Geometry.get_elements(
        get_geometry(get_space(geometry)), patch_id, local_object_id, geometric_dim
    )
end

function Geometry.evaluate(
    geometry::FEGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    space = get_space(geometry)
    coefficients = get_coefficients(geometry)

    return mapreduce(
        c -> evaluate(space, element_id, xi, 0, c)[1][1][1], hcat, eachcol(coefficients)
    )
end

# function Geometry.jacobian(
#     mapping::FEMapping{manifold_dim, image_dim}, x::Matrix{Float64}
# ) where {manifold_dim, image_dim}
#     num_points = size(x, 1)

#     return [
#         SMatrix{image_dim, manifold_dim}(mapping.dmapping(view(x, i, :))) for
#         i in 1:num_points
#     ]
# end

# function Geometry.hessian(
#     mapping::FEMapping{manifold_dim, image_dim}, x::Matrix{Float64}
# ) where {manifold_dim, image_dim}
#     return [
#         ntuple(image_dim) do i
#             return SMatrix{manifold_dim, manifold_dim}(mapping.ddmapping(view(x, p, :))[i])
#         end for p in axes(x, 1)
#     ]
# end

function DiscreteGeometry(
    space::S, coefficients::Matrix{T}
) where {
    manifold_dim, num_patches, S <: AbstractFESpace{manifold_dim, 1, num_patches}, T <: Real
}
    image_dim = size(coefficients, 2)
    num_elements = get_num_elements(space)
    num_elements_per_patch = get_num_elements_per_patch(space)
    function evaluable_function(
        element_id::Int, xi::Points.AbstractPoints{manifold_dim}, num_derivatives::Int=0
    )
        space_basis, basis_indices = evaluate(space, element_id, xi, num_derivatives)
        eval = Vector{Vector{Matrix{Float64}}}(undef, num_derivatives + 1) # 1 component
        num_points = Points.get_num_points(xi)
        for i in eachindex(eval)
            num_ders = length(space_basis[i])
            eval[i] = Vector{Matrix{Float64}}(undef, num_ders)
            for j in eachindex(eval[i])
                eval[i][j] = Matrix{Float64}(undef, num_points, image_dim)
                for l in 1:image_dim
                    eval[i][j][:, l] = space_basis[i][j][1] * coefficients[basis_indices, l]
                end
            end
        end

        return eval
    end

    function element_length_function(element_id::Int)
        return Geometry.get_element_lengths(get_geometry(space), element_id)
    end

    return Geometry.DiscreteGeometry(
        manifold_dim,
        image_dim,
        num_elements,
        evaluable_function,
        element_length_function,
        num_elements_per_patch,
    )
end

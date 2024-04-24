struct FEMGeometry{n,m} <: AbstractGeometry
    geometry_coeffs::Array{Float64,2}
    fem_space::AbstractFiniteElementSpace{n}
    n_elements::Int

    function FEMGeometry(fem_space::AbstractFiniteElementSpace{n}, geometry_coeffs::Array{Float64,2}) where {n}
        m = size(geometry_coeffs,2)
        n_elements = get_num_elements(fem_space)
        return new{n,m}(geometry_coeffs, fem_space, n_elements)
    end
end

function evaluate(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    # evaluate fem space
    fem_basis, fem_basis_indices, _ = evaluate(geometry.fem_space, element_id, xi, 0)
    # combine with coefficients and return
    return fem_basis * geometry.geometry_coeffs[fem_basis_indices,:]
end

function jacobian(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    # evaluate fem space
    fem_basis, fem_basis_indices, _ = evaluate(geometry.fem_space, element_id, xi, 1)
    # combine with coefficients and return
    return (fem_basis[:,:,k+1] * geometry.geometry_coeffs[fem_basis_indices,:] for k = 1:n), fem_basis[:,:,1] * geometry.geometry_coeffs[fem_basis_indices,:]
end
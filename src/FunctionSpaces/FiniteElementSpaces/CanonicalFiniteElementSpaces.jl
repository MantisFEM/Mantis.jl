using LinearAlgebra
using SparseArrays

"""
    CanonicalFiniteElementSpace{C} <: AbstractFiniteElementSpace{1}

Wrapper that allows treating a canonical space as a finite element space.

# Fields
- `canonical_space::C`: The underlying canonical space.
- `dof_partition::Vector{Vector{Int}}`: Partition of degrees of freedom.

# Constructor
    CanonicalFiniteElementSpace(canonical_space::C) where C <: AbstractCanonicalSpace

Constructs a `CanonicalFiniteElementSpace` from a given canonical space.
"""
struct CanonicalFiniteElementSpace{C} <: AbstractFiniteElementSpace{1}
    canonical_space::C
    dof_partition::Vector{Vector{Int}}

    function CanonicalFiniteElementSpace(canonical_space::C) where {C <: AbstractCanonicalSpace}
        CanonicalFiniteElementSpace(canonical_space, 1, 1)
    end

    function CanonicalFiniteElementSpace(canonical_space::C, n_dofs_left::Int, n_dofs_right::Int) where {C <: AbstractCanonicalSpace}
        # Allocate memory for degree of freedom partitioning
        dof_partition = Vector{Vector{Int}}(undef,3)
        # First, store the left dofs ...
        dof_partition[1] = collect(1:n_dofs_left)
        # ... then the interior dofs ...
        dof_partition[2] = collect(n_dofs_left+1:canonical_space.p-n_dofs_right+1)
        # ... and then finally the right dofs.
        dof_partition[3] = collect(canonical_space.p-n_dofs_right+2:canonical_space.p+1)
        
        new{C}(canonical_space, dof_partition)
    end
end

"""
    get_polynomial_degree(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}

Get the polynomial degree of the canonical finite element space.

# Arguments
- `space::CanonicalFiniteElementSpace{C}`: The canonical finite element space.

# Returns
The polynomial degree of the space.
"""
function get_polynomial_degree(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}
    return space.canonical_space.p
end

"""
    get_polynomial_degree(space::CanonicalFiniteElementSpace{C}, ::Int) where {C<: AbstractCanonicalSpace}

Get the polynomial degree of the canonical finite element space, ignoring the second argument.

# Arguments
- `space::CanonicalFiniteElementSpace{C}`: The canonical finite element space.
- `::Int`: Unused argument (placeholder).

# Returns
The polynomial degree of the space.
"""
function get_polynomial_degree(space::CanonicalFiniteElementSpace{C}, ::Int) where {C<: AbstractCanonicalSpace}
    return space.canonical_space.p
end

"""
    get_num_elements(::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}

Get the number of elements in the canonical finite element space.

# Arguments
- `::CanonicalFiniteElementSpace{C}`: The canonical finite element space.

# Returns
Always returns 1, as canonical spaces have a single element.
"""
function get_num_elements(::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}
    return 1
end

"""
    get_num_basis(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}

Get the dimension of the canonical finite element space.

# Arguments
- `space::CanonicalFiniteElementSpace{C}`: The canonical finite element space.

# Returns
The dimension of the space, which is polynomial degree + 1.
"""
function get_num_basis(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}
    return space.canonical_space.p + 1
end

"""
    _evaluate_all_at_point(space::CanonicalFiniteElementSpace{C}, xi::Float64, nderivatives::Int) where {C<: AbstractCanonicalSpace}

Evaluate all basis functions and their derivatives at a given point.

# Arguments
- `space::CanonicalFiniteElementSpace{C}`: The canonical finite element space.
- `xi::Float64`: The point at which to evaluate the basis functions.
- `nderivatives::Int`: The number of derivatives to compute.

# Returns
A sparse matrix containing the values of the basis functions and their derivatives.
"""
function _evaluate_all_at_point(space::CanonicalFiniteElementSpace{C}, xi::Float64, nderivatives::Int) where {C<: AbstractCanonicalSpace}
    local_basis = evaluate(space.canonical_space, [xi], nderivatives)
    ndofs = get_num_basis(space)
    basis_indices = 1:ndofs
    nloc = length(basis_indices)
    I = zeros(Int, nloc * (nderivatives + 1))
    J = zeros(Int, nloc * (nderivatives + 1))
    V = zeros(Float64, nloc * (nderivatives + 1))
    count = 0
    for r = 0:nderivatives
        for i = 1:nloc
            I[count+1] = basis_indices[i]
            J[count+1] = r+1
            V[count+1] = local_basis[r+1][1][1, i]
            count += 1
        end
    end
    return SparseArrays.sparse(I,J,V,ndofs,nderivatives+1)
end

"""
    _evaluate_all_at_point(space::CanonicalFiniteElementSpace{C}, ::Int, xi::Float64, nderivatives::Int) where {C<: AbstractCanonicalSpace}

Evaluate all basis functions and their derivatives at a given point, ignoring the second argument.

# Arguments
- `space::CanonicalFiniteElementSpace{C}`: The canonical finite element space.
- `::Int`: Unused argument (placeholder).
- `xi::Float64`: The point at which to evaluate the basis functions.
- `nderivatives::Int`: The number of derivatives to compute.

# Returns
A sparse matrix containing the values of the basis functions and their derivatives.
"""
function _evaluate_all_at_point(space::CanonicalFiniteElementSpace{C}, ::Int, xi::Float64, nderivatives::Int) where {C<: AbstractCanonicalSpace}
    return _evaluate_all_at_point(space, xi, nderivatives)
end

"""
    get_local_basis(space::CanonicalFiniteElementSpace{C}, ::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int) where {C<: AbstractCanonicalSpace}

Get the local basis functions and their derivatives at given points.

# Arguments
- `space::CanonicalFiniteElementSpace{C}`: The canonical finite element space.
- `::Int`: Unused argument (placeholder).
- `xi::NTuple{1,Vector{Float64}}`: The points at which to evaluate the basis functions.
- `nderivatives::Int`: The number of derivatives to compute.

# Returns
The local basis functions and their derivatives evaluated at the given points.
"""
function get_local_basis(space::CanonicalFiniteElementSpace{C}, ::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int) where {C<: AbstractCanonicalSpace}
    return evaluate(space.canonical_space, xi[1], nderivatives)
end

"""
    get_basis_indices(space::CanonicalFiniteElementSpace{C}, ::Int) where {C<: AbstractCanonicalSpace}

Get the indices of the basis functions.

# Arguments
- `space::CanonicalFiniteElementSpace{C}`: The canonical finite element space.
- `::Int`: Unused argument (placeholder).

# Returns
A vector of indices for the basis functions.
"""
function get_basis_indices(space::CanonicalFiniteElementSpace{C}, ::Int) where {C<: AbstractCanonicalSpace}
    return collect(1:get_num_basis(space))
end

"""
    get_max_local_dim(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}

Get the maximum local dimension of the canonical finite element space.

# Arguments
- `space::CanonicalFiniteElementSpace{C}`: The canonical finite element space.

# Returns
The maximum local dimension, which is equal to the total dimension for canonical spaces.
"""
function get_max_local_dim(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}
    return get_num_basis(space)
end

"""
    get_dof_partition(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}

Get the partition of degrees of freedom for the canonical finite element space.

# Arguments
- `space::CanonicalFiniteElementSpace{C}`: The canonical finite element space.

# Returns
The partition of degrees of freedom.
"""
function get_dof_partition(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}
    return space.dof_partition
end

"""
    get_extraction(space::CanonicalFiniteElementSpace{C}, ::Int) where {C<: AbstractCanonicalSpace}

Get the extraction operator for the canonical finite element space.

# Arguments
- `space::CanonicalFiniteElementSpace{C}`: The canonical finite element space.
- `::Int`: Unused argument (placeholder).

# Returns
A tuple containing the extraction operator (identity matrix) and the basis indices.
"""
function get_extraction(space::CanonicalFiniteElementSpace{C}, ::Int) where {C<: AbstractCanonicalSpace}
    basis_indices = get_basis_indices(space, 1)
    nbasis = length(basis_indices)
    return Matrix(LinearAlgebra.I, nbasis, nbasis), basis_indices
end

"""
    evaluate(space::CanonicalFiniteElementSpace{C}, ::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int) where {C<: AbstractCanonicalSpace}

Evaluate the basis functions and their derivatives at given points.

# Arguments
- `space::CanonicalFiniteElementSpace{C}`: The canonical finite element space.
- `::Int`: Unused argument (placeholder).
- `xi::NTuple{1,Vector{Float64}}`: The points at which to evaluate the basis functions.
- `nderivatives::Int`: The number of derivatives to compute.

# Returns
A tuple containing the local basis functions and their derivatives, and the basis indices.
"""
function evaluate(space::CanonicalFiniteElementSpace{C}, ::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int) where {C<: AbstractCanonicalSpace}
    local_basis = get_local_basis(space, 1, xi, nderivatives)
    basis_indices = get_basis_indices(space, 1)

    return local_basis, basis_indices
end
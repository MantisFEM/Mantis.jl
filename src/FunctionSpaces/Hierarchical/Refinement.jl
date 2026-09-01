############################################################################################
#                                         Uniform                                          #
############################################################################################

function refinement_uniform(
    parent::AbstractFESpace{manifold_dim}, num_subdivisions::NTuple{manifold_dim, Int}
) where {manifold_dim}
    return throw(MethodError(refinement_uniform, (parent, num_subdivisions)))
end

function refinement_uniform(
    parent::BSplineSpace, num_subdivisions::Int, child_multiplicity::Int=1
)
    child_knot_vector = refinement_uniform(
        get_knot_vector(parent), num_subdivisions, child_multiplicity
    )
    child_parametric_geometry = get_geometry(child_knot_vector)
    child_geometry = Geometry.refinement_uniform(get_geometry(parent), num_subdivisions)
    child_polynomials = get_child_canonical_space(get_polynomials(parent), num_subdivisions)
    dof_partition = get_dof_partition(parent)
    n_dofs_left = length(dof_partition[1][1])
    n_dofs_right = length(dof_partition[1][3])
    p = get_polynomial_degree(parent)

    return BSplineSpace(
        child_geometry,
        child_parametric_geometry,
        child_polynomials,
        p .- get_multiplicity(child_knot_vector),
        n_dofs_left,
        n_dofs_right,
    )
end

function refinement_uniform(
    parent::KnotVector, num_subdivisions::Int, child_multiplicity::Int=1
)
    child_geometry = Geometry.refinement_uniform(get_geometry(parent), num_subdivisions)
    child_multiplicity_vector = refinement_uniform(
        get_multiplicity(parent), num_subdivisions, child_multiplicity
    )
    p = get_polynomial_degree(parent)

    return KnotVector(child_geometry, p, child_multiplicity_vector)
end

function refinement_uniform(
    parent::Vector{Int}, num_subdivisions::Int, child_multiplicity::Int=1
)
    mult_length = 1 + (length(parent) - 1) * num_subdivisions
    child_multiplicity_vector = fill(child_multiplicity, mult_length)
    for i in eachindex(parent)
        j = 1 + (i - 1) * num_subdivisions
        pj = parent[i]
        child_multiplicity_vector[j] = max(pj, child_multiplicity)
    end

    return child_multiplicity_vector
end

function refinement_uniform(
    parent::TensorProductSpace{manifold_dim}, num_subdivisions::NTuple{manifold_dim, Int}
) where {manifold_dim}
    parametric_geometry = Geometry.refinement_uniform(
        get_parametric_geometry(parent), num_subdivisions
    )
    geometry = Geometry.refinement_uniform(get_geometry(parent), num_subdivisions)
    factors = get_factor_spaces(parent)
    child_factors = map(refinement_uniform, factors, num_subdivisions)

    return TensorProductSpace(child_factors, geometry, parametric_geometry)
end

function refinement_uniform(
    parent::GTBSplineSpace{num_patches}, num_subdivisions::Int
) where {num_patches}
    return refinement_uniform(parent, ntuple(_ -> num_subdivisions, num_patches))
end

function refinement_uniform(
    parent::GTBSplineSpace{num_patches}, num_subdivisions::NTuple{num_patches, Int}
) where {num_patches}
    child_patch_spaces = map(refinement_uniform, get_patch_spaces(parent), num_subdivisions)
    child = GTBSplineSpace(
        child_patch_spaces,
        get_patch_interface_regularities(parent),
        get_constructor_num_dofs_left(parent),
        get_constructor_num_dofs_right(parent),
    )

    return child
end

############################################################################################
#                                     Degree-Elevation                                     #
############################################################################################

function refinement_degree(
    parent::AbstractFESpace{manifold_dim}, degree_delta::Int
) where {manifold_dim}
    return throw(MethodError(refinement_degree, (parent, degree_delta)))
end

function refinement_degree(parent::KnotVector, degree_delta::Int)
    iszero(degree_delta) && return deepcopy(parent)

    child = KnotVector(
        get_geometry(parent),
        get_polynomial_degree(parent) + degree_delta,
        get_multiplicity(parent) .+ degree_delta,
    )

    return child
end

function refinement_degree(parent::BSplineSpace, degree_delta::Int)
    iszero(degree_delta) && return deepcopy(parent)

    child_knot_vector = refinement_degree(get_knot_vector(parent), degree_delta)
    child_polynomials = refinement_degree(get_polynomials(parent), degree_delta)
    child_degree = get_polynomial_degree(child_knot_vector)
    child_regularity = child_degree .- get_multiplicity(child_knot_vector)
    dof_partition = get_dof_partition(parent)
    n_dofs_left = length(dof_partition[1][1])
    n_dofs_right = length(dof_partition[1][3])

    return BSplineSpace(
        get_geometry(parent),
        get_geometry(child_knot_vector),
        child_polynomials,
        child_regularity,
        n_dofs_left,
        n_dofs_right,
    )
end

function refinement_degree(parent::Bernstein, degree_delta::Int)
    return Bernstein(get_polynomial_degree(parent) + degree_delta)
end

############################################################################################
#                                      TensorProduct                                       #
############################################################################################

function Hierarchical.Refinement(
    parent::TensorProductSpace, methods::NTuple{num_methods, Function}
) where {num_methods}
    function refinement(parent)
        parent_tp = get_tensor_product(parent)
        child_tp = Hierarchical.Refinement(parent_tp, methods)()

        return TensorProductSpace(TensorProducts.get_factors(child_tp))
    end

    return Hierarchical.Refinement(parent, refinement)
end

function Hierarchical.Refinement(
    parent::TensorProductSpace,
    geometry::Geometry.AbstractGeometry,
    parametric_geometry::Geometry.AbstractGeometry,
    methods::NTuple{num_methods, Function},
) where {num_methods}
    function refinement(parent)
        parent_tp = get_tensor_product(parent)
        child_tp = Hierarchical.Refinement(parent_tp, methods)()
        return TensorProductSpace(
            TensorProducts.get_factors(child_tp), geometry, parametric_geometry
        )
    end

    return Hierarchical.Refinement(parent, refinement)
end

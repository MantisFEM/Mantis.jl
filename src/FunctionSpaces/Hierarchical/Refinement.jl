############################################################################################
#                                         Uniform                                          #
############################################################################################

"""
    refinement_uniform(
        parent::AbstractFESpace{manifold_dim},
        num_subdivisions::NTuple{manifold_dim, Int},
    ) where {manifold_dim}

Return a uniformly refined space of the same type as `parent`, subdividing each element
according to `num_subdivisions`.
"""
function refinement_uniform(
    parent::AbstractFESpace{manifold_dim}, num_subdivisions::NTuple{manifold_dim, Int}
) where {manifold_dim}
    return throw(MethodError(refinement_uniform, (parent, num_subdivisions)))
end

function refinement_uniform(parent::AbstractFESpace{1}, num_subdivisions::NTuple{1, Int})
    return refinement_uniform(parent, first(num_subdivisions))
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
    child_patch_spaces = map(
        refinement_uniform,
        get_patch_spaces(parent),
        ntuple(_ -> num_subdivisions, num_patches),
    )
    child = GTBSplineSpace(
        child_patch_spaces,
        get_patch_interface_regularities(parent),
        get_constructor_num_dofs_left(parent),
        get_constructor_num_dofs_right(parent),
    )

    return child
end

function refinement_uniform(parent::PolarSplineSpace, num_subdivisions::NTuple{2, Int})
    degen_cp_parent = get_degenerate_control_points(parent)
    size_degen_cp_parent = size(degen_cp_parent)
    degen_scal = scaling_uniform(get_degenerate_space(parent), num_subdivisions)
    degen_space_child = get_child(degen_scal)
    subdiv_mat_tp = Hierarchical.get_scaling_matrix(degen_scal)
    size_tp_child = map(get_num_basis, get_factor_spaces(degen_space_child))
    degen_cp_child = reshape(
        subdiv_mat_tp * reshape(degen_cp_parent, :, size_degen_cp_parent[3]),
        (size_tp_child[1], size_tp_child[2], size_degen_cp_parent[3]),
    )
    patch_spaces_child = map(
        ps -> refinement_uniform(ps, num_subdivisions), get_patch_spaces(parent)
    )

    return PolarSplineSpace(
        patch_spaces_child,
        degen_cp_child,
        degen_space_child,
        parent.two_poles,
        parent.zero_at_poles,
    )
end

function refinement_uniform(
    parent::DirectSumSpace{manifold_dim, num_components},
    num_subdivions::NTuple{num_components, NTuple{manifold_dim, Int}},
) where {manifold_dim, num_components}
    child_components = map(refinement_uniform, get_component_spaces(parent), num_subdivions)

    return DirectSumSpace(child_components)
end

function refinement_uniform(
    parent::DirectSumSpace{manifold_dim, num_components},
    num_subdivions::NTuple{manifold_dim, Int},
) where {manifold_dim, num_components}
    return refinement_uniform(parent, ntuple(_ -> num_subdivions, num_components))
end

############################################################################################
#                                     Degree-Elevation                                     #
############################################################################################

"""
    refinement_degree(
        parent::AbstractFESpace{manifold_dim}, degree_delta::NTuple{manifold_dim, Int}
    ) where {manifold_dim}

Return the space obtained by increasing the polynomial degree of `parent` by `degree_delta`.
"""
function refinement_degree(
    parent::AbstractFESpace{manifold_dim}, degree_delta::NTuple{manifold_dim, Int}
) where {manifold_dim}
    return throw(MethodError(refinement_degree, (parent, degree_delta)))
end

function refinement_degree(parent::AbstractFESpace{1}, degree_delta::NTuple{1, Int})
    return refinement_degree(parent, first(degree_delta))
end

function refinement_degree(parent::KnotVector, degree_delta::Int)
    # MUTABILITY: We return a deepcopy here to avoid having the output share state with the
    # input. Otherwise, it would lead to inconsistent behaviour, since `degree_delta!=0`
    # returns a new object.
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
#                                 Hierarchical.Refinement                                  #
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

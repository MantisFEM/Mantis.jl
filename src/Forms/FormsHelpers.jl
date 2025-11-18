############################################################################################
#                                    Form construction                                     #
############################################################################################
function build_form_field(
    form_space::AbstractFormSpace; label::Union{String, Nothing}=nothing
)
    if isnothing(label)
        return FormField(form_space, form_space.label)
    end

    return FormField(form_space, label)
end

function build_form_field(
    form_space::AbstractFormSpace,
    coeffs::Vector{Float64};
    label::Union{String, Nothing}=nothing,
)
    if length(coeffs) != get_num_basis(form_space)
        throw(
            ArgumentError("""\
                Size mismatch. Number of given coefficients differs from the dimension of\
                `form_space`. The given numbers were $(length(coeffs)) and \
                $(get_num_basis(form_space)), respectively.
                """)
        )
    end
    form_field = build_form_field(form_space; label)
    form_field.coefficients .= coeffs

    return form_field
end

function build_form_fields(
    form_spaces::FS; labels::Union{L, Nothing}=nothing
) where {
    num_forms, FS <: NTuple{num_forms, AbstractFormSpace}, L <: NTuple{num_forms, String}
}
    if isnothing(labels)
        labels = ntuple(num_forms) do _
            return nothing
        end
    end

    form_fields = ntuple(num_forms) do i
        return build_form_field(form_spaces[i]; label=labels[i])
    end

    return form_fields
end

function build_form_fields(
    form_spaces::FS, coeffs::Vector{Float64}; labels::Union{L, Nothing}=nothing
) where {
    num_forms, FS <: NTuple{num_forms, AbstractFormSpace}, L <: NTuple{num_forms, String}
}
    if isnothing(labels)
        labels = ntuple(num_forms) do _
            return nothing
        end
    end

    start_id = 1
    form_fields = ntuple(num_forms) do i
        num_coeffs = get_num_basis(form_spaces[i])
        ff = build_form_field(
            form_spaces[i],
            coeffs[start_id:(start_id + num_coeffs - 1)];
            label=labels[i],
        )
        start_id += num_coeffs

        return ff
    end

    return form_fields
end

################################################################################
# Tensor-product B-spline de Rham complex
################################################################################

import Combinatorics

"""
    get_basis_index_combinations(manifold_dim::Int, form_rank::Int)

Generate all possible k-form basis index combinations.

# Arguments
- `manifold_dim::Int`: the dimension of the manifold.
- `form_rank::Int`: the rank of the form.

# Returns
- `NTuple{binomial(manifold_dim, form_rank), Vector{Int}}`: the basis index combinations.
"""
function get_basis_index_combinations(manifold_dim::Int, form_rank::Int)

    # generate all possible k-form basis index combinations
    if form_rank == 2 && manifold_dim == 3
        # canonical ordering of the degree-deficit combinations
        return ([2, 3], [1, 3], [1, 2])::NTuple{3, Vector{Int}}
    else
        return tuple(
            Combinatorics.combinations(1:manifold_dim, form_rank)...
        )::NTuple{binomial(manifold_dim, form_rank), Vector{Int}}
    end
end

"""
    create_tensor_product_bspline_de_rham_complex(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        section_spaces::NTuple{manifold_dim, F},
        regularities::NTuple{manifold_dim, Int},
        geometry::G,
    ) where {
        manifold_dim,
        F <: FunctionSpaces.AbstractCanonicalSpace,
        G <: Geometry.AbstractGeometry{manifold_dim},
    }

Create a tensor-product B-spline de Rham complex.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: the starting points of the domain.
- `box_sizes::NTuple{manifold_dim, Float64}`: the sizes of the domain.
- `num_elements::NTuple{manifold_dim, Int}`: the number of elements in each direction.
- `section_spaces::NTuple{manifold_dim, F}`: the section spaces.
- `regularities::NTuple{manifold_dim, Int}`: the regularities of the B-spline spaces.
- `geometry::G`: the geometry of the domain.

# Returns
- `::Tuple{<:AbstractFormSpace{manifold_dim, form_rank, geometry}}`: Tuple with the form
    spaces of the complex, for each `form_rank` from `0` to `manifold_dim+1`.
"""
function create_tensor_product_bspline_de_rham_complex(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    section_spaces::NTuple{manifold_dim, F},
    regularities::NTuple{manifold_dim, Int},
    geometry::G,
) where {
    manifold_dim,
    F <: FunctionSpaces.AbstractCanonicalSpace,
    G <: Geometry.AbstractGeometry{manifold_dim},
}

    # number of dofs on the left and right boundary of the domain
    n_dofs_left = tuple((1 for _ in 1:manifold_dim)...)
    n_dofs_right = tuple((1 for _ in 1:manifold_dim)...)

    # store all univariate FEM spaces helper
    fem_spaces = Vector{NTuple{manifold_dim, FunctionSpaces.AbstractFESpace{1, 1}}}(
        undef, 2
    )
    # first, create all univariate FEM spaces corresponding to directional-zero forms
    fem_spaces[1] = FunctionSpaces.create_dim_wise_bspline_spaces(
        starting_points,
        box_sizes,
        num_elements,
        section_spaces,
        regularities,
        n_dofs_left,
        n_dofs_right,
    )
    # next, create all univariate FEM spaces corresponding to directional-one forms
    fem_spaces[2] = map(FunctionSpaces.get_derivative_space, fem_spaces[1])

    # Build all the form spaces of the complex.
    form_spaces = ntuple(manifold_dim + 1) do k
        k = k - 1 # Because form ranks range from 0 to manifold_dim.
        # Get k-form basis indices, these also inform the directional degree-deficits.
        k_form_basis_idxs = get_basis_index_combinations(manifold_dim, k)
        num_form_components = length(k_form_basis_idxs)
        # Generate tuple with all the k-form finite element spaces.
        k_form_fem_spaces = ntuple(num_form_components) do component
            # By default, use direction-zero forms...
            fem_space_idxs = ones(Int, manifold_dim)
            # ...unless the basis index is present in the k-form basis indices.
            fem_space_idxs[k_form_basis_idxs[component]] .= 2
            # Build and store constituent spaces of the tensor-product FEM space.
            tp_consituent_spaces = ntuple(manifold_dim) do dim
                return fem_spaces[fem_space_idxs[dim]][dim]
            end
            # Build and return the corresponding tensor-product FEM space.
            return FunctionSpaces.TensorProductSpace(tp_consituent_spaces)
        end
        if num_form_components == 1
            return FormSpace(k, geometry, k_form_fem_spaces[1], "ω_$k")
        else
            return FormSpace(
                k, geometry, FunctionSpaces.DirectSumSpace(k_form_fem_spaces), "ω_$k"
            )
        end
    end

    return form_spaces
end

"""
    create_tensor_product_bspline_de_rham_complex(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        section_spaces::NTuple{manifold_dim, F},
        regularities::NTuple{manifold_dim, Int},
    ) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace}

Create a tensor-product B-spline de Rham complex.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: the starting points of the domain.
- `box_sizes::NTuple{manifold_dim, Float64}`: the sizes of the domain.
- `num_elements::NTuple{manifold_dim, Int}`: the number of elements in each direction.
- `section_spaces::NTuple{manifold_dim, F}`: the section spaces.
- `regularities::NTuple{manifold_dim, Int}`: the regularities of the B-spline spaces.

# Returns
- `Vector{AbstractFormSpace}`: the `manifold_dim+1` form spaces of the complex.
"""
function create_tensor_product_bspline_de_rham_complex(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    section_spaces::NTuple{manifold_dim, F},
    regularities::NTuple{manifold_dim, Int},
) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace}

    # create underlying box geometry
    geometry = Geometry.create_cartesian_box(starting_points, box_sizes, num_elements)

    return create_tensor_product_bspline_de_rham_complex(
        starting_points, box_sizes, num_elements, section_spaces, regularities, geometry
    )
end

"""
    create_tensor_product_bspline_de_rham_complex(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        degrees::NTuple{manifold_dim, Int},
        regularities::NTuple{manifold_dim, Int},
    ) where {manifold_dim}

Create a tensor-product B-spline de Rham complex.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: the starting points of the domain.
- `box_sizes::NTuple{manifold_dim, Float64}`: the sizes of the domain.
- `num_elements::NTuple{manifold_dim, Int}`: the number of elements in each direction.
- `degrees::NTuple{manifold_dim, Int}`: the degrees of the B-spline spaces.
- `regularities::NTuple{manifold_dim, Int}`: the regularities of the B-spline spaces.

# Returns
- `Vector{AbstractFormSpace}`: the `manifold_dim+1` form spaces of the complex.
"""
function create_tensor_product_bspline_de_rham_complex(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    degrees::NTuple{manifold_dim, Int},
    regularities::NTuple{manifold_dim, Int},
) where {manifold_dim}
    return create_tensor_product_bspline_de_rham_complex(
        starting_points,
        box_sizes,
        num_elements,
        map(FunctionSpaces.Bernstein, degrees),
        regularities,
    )
end

"""
    create_curvilinear_tensor_product_bspline_de_rham_complex(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        section_spaces::NTuple{manifold_dim, F},
        regularities::NTuple{manifold_dim, Int},
    ) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace}

Create a tensor-product B-spline de Rham complex on a crazy mesh.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: the starting points of the domain.
- `box_sizes::NTuple{manifold_dim, Float64}`: the sizes of the domain.
- `num_elements::NTuple{manifold_dim, Int}`: the number of elements in each direction.
- `section_spaces::NTuple{manifold_dim, F}`: the section spaces.
- `regularities::NTuple{manifold_dim, Int}`: the regularities of the B-spline spaces.

# Returns
- `Vector{AbstractFormSpace}`: the `manifold_dim+1` form spaces of the complex.
"""
function create_curvilinear_tensor_product_bspline_de_rham_complex(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    section_spaces::NTuple{manifold_dim, F},
    regularities::NTuple{manifold_dim, Int};
    crazy_c::Float64=0.1,
) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace}

    # create underlying box geometry
    geometry = Geometry.create_curvilinear_square(
        starting_points, box_sizes, num_elements; crazy_c=crazy_c
    )

    return create_tensor_product_bspline_de_rham_complex(
        starting_points, box_sizes, num_elements, section_spaces, regularities, geometry
    )
end

# TODO: These methods should work for more general mappings.

"""
    create_curvilinear_tensor_product_bspline_de_rham_complex(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        degrees::NTuple{manifold_dim, Int},
        regularities::NTuple{manifold_dim, Int},
    ) where {manifold_dim}

Create a tensor-product B-spline de Rham complex on a crazy geometry.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: the starting points of the domain.
- `box_sizes::NTuple{manifold_dim, Float64}`: the sizes of the domain.
- `num_elements::NTuple{manifold_dim, Int}`: the number of elements in each direction.
- `degrees::NTuple{manifold_dim, Int}`: the degrees of the B-spline spaces.
- `regularities::NTuple{manifold_dim, Int}`: the regularities of the B-spline spaces.

# Returns
- `Vector{AbstractFormSpace}`: the `manifold_dim+1` form spaces of the complex.
"""
function create_curvilinear_tensor_product_bspline_de_rham_complex(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    degrees::NTuple{manifold_dim, Int},
    regularities::NTuple{manifold_dim, Int};
    crazy_c::Float64=0.1,
) where {manifold_dim}
    return create_curvilinear_tensor_product_bspline_de_rham_complex(
        starting_points,
        box_sizes,
        num_elements,
        map(FunctionSpaces.Bernstein, degrees),
        regularities;
        crazy_c=crazy_c,
    )
end

############################################################################################
#                               Hierarchical de Rham complex                               #
############################################################################################

function create_hierarchical_de_rham_complex(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    section_spaces::NTuple{manifold_dim, F},
    regularities::NTuple{manifold_dim, Int},
    num_subdivisions::NTuple{manifold_dim, Int},
    truncate::Bool,
    simplified::Bool,
    geometry::G,
) where {
    manifold_dim,
    F <: FunctionSpaces.AbstractCanonicalSpace,
    G <: Geometry.AbstractGeometry{manifold_dim},
}
    # number of dofs on the left and right boundary of the domain
    n_dofs_left = tuple((1 for _ in 1:manifold_dim)...)
    n_dofs_right = tuple((1 for _ in 1:manifold_dim)...)

    # store all univariate FEM spaces helper
    fem_spaces = Vector{NTuple{manifold_dim, FunctionSpaces.AbstractFESpace{1, 1}}}(
        undef, 2
    )
    # first, create all univariate FEM spaces corresponding to directional-zero forms
    fem_spaces[1] = FunctionSpaces.create_dim_wise_bspline_spaces(
        starting_points,
        box_sizes,
        num_elements,
        section_spaces,
        regularities,
        n_dofs_left,
        n_dofs_right,
    )
    # next, create all univariate FEM spaces corresponding to directional-one forms
    fem_spaces[2] = map(FunctionSpaces.get_derivative_space, fem_spaces[1])

    # Build all the form spaces of the complex.
    form_spaces = ntuple(manifold_dim + 1) do k
        k = k - 1 # Because form ranks range from 0 to manifold_dim.
        # Get k-form basis indices, these also inform the directional degree-deficits.
        k_form_basis_idxs = get_basis_index_combinations(manifold_dim, k)
        num_form_components = length(k_form_basis_idxs)
        # Generate tuple with all the k-form finite element spaces.
        k_form_fem_spaces = ntuple(num_form_components) do component
            # By default, use direction-zero forms...
            fem_space_idxs = ones(Int, manifold_dim)
            # ...unless the basis index is present in the k-form basis indices.
            fem_space_idxs[k_form_basis_idxs[component]] .= 2
            # Build and store constituent spaces of the tensor-product FEM space.
            tp_consituent_spaces = ntuple(manifold_dim) do dim
                return fem_spaces[fem_space_idxs[dim]][dim]
            end
            # Build the corresponding tensor-product FEM space.
            tp_space = FunctionSpaces.TensorProductSpace(tp_consituent_spaces)
            hierarchical_space = FunctionSpaces.HierarchicalFiniteElementSpace(
                [tp_space],
                FunctionSpaces.AbstractTwoScaleOperator[],
                [Int[]],
                num_subdivisions,
                truncate,
                simplified,
            )

            return hierarchical_space
        end

        if num_form_components == 1
            return FormSpace(k, geometry, k_form_fem_spaces[1], "ω_$k")
        else
            return FormSpace(
                k, geometry, FunctionSpaces.DirectSumSpace(k_form_fem_spaces), "ω_$k"
            )
        end
    end

    return form_spaces
end

function create_hierarchical_de_rham_complex(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    degrees::NTuple{manifold_dim, Int},
    regularities::NTuple{manifold_dim, Int},
    num_subdivisions::NTuple{manifold_dim, Int},
    truncate::Bool,
    simplified::Bool,
) where {manifold_dim}

    # create underlying box geometry
    geometry = Geometry.create_cartesian_box(starting_points, box_sizes, num_elements)

    return create_hierarchical_de_rham_complex(
        starting_points,
        box_sizes,
        num_elements,
        map(FunctionSpaces.Bernstein, degrees),
        regularities,
        num_subdivisions,
        truncate,
        simplified,
        geometry,
    )
end

function update_hierarchical_de_rham_complex(
    complex::C,
    refinement_domains::Vector{Vector{Int}},
    new_zero_form_ts::FunctionSpaces.AbstractTwoScaleOperator{2},
    new_zero_form_space::FunctionSpaces.AbstractFESpace{2, 1},
) where {num_forms, C <: NTuple{num_forms, AbstractFormSpace}}
    zero_form_space = FunctionSpaces.get_component_spaces(complex[1].fem_space)[1]
    one_form_space_x, one_form_space_y = FunctionSpaces.get_component_spaces(
        complex[2].fem_space
    )
    two_form_space = FunctionSpaces.get_component_spaces(complex[3].fem_space)[1]

    num_sub = FunctionSpaces.get_num_subdivisions(zero_form_space)
    L = FunctionSpaces.get_num_levels(zero_form_space)

    # Build refined 0-form space, get domains and geometry
    zero_form_space = FunctionSpaces.update_hierarchical_space!(
        zero_form_space, refinement_domains, new_zero_form_ts, new_zero_form_space
    )
    zero_form_space_domains = FunctionSpaces.get_level_ids(zero_form_space.nested_domains)

    # Update other k-forms using 0-form domains
    new_one_form_x_ts, new_one_form_x_space = FunctionSpaces.build_two_scale_operator(
        FunctionSpaces.get_space(one_form_space_x, L), num_sub
    )
    new_one_form_y_ts, new_one_form_y_space = FunctionSpaces.build_two_scale_operator(
        FunctionSpaces.get_space(one_form_space_y, L), num_sub
    )
    new_two_form_ts, new_two_form_space = FunctionSpaces.build_two_scale_operator(
        FunctionSpaces.get_space(two_form_space, L), num_sub
    )

    one_form_space_x = FunctionSpaces.update_hierarchical_space!(
        one_form_space_x, zero_form_space_domains, new_one_form_x_ts, new_one_form_x_space
    )
    one_form_space_y = FunctionSpaces.update_hierarchical_space!(
        one_form_space_y, zero_form_space_domains, new_one_form_y_ts, new_one_form_y_space
    )
    two_form_space = FunctionSpaces.update_hierarchical_space!(
        two_form_space, zero_form_space_domains, new_two_form_ts, new_two_form_space
    )

    # Build new form spaces
    geom = Geometry.HierarchicalGeometry(zero_form_space)
    one_form_space_sum = FunctionSpaces.DirectSumSpace((one_form_space_x, one_form_space_y))
    zero_form = Forms.FormSpace(0, geom, zero_form_space, "β")
    one_form = Forms.FormSpace(1, geom, one_form_space_sum, "σ")
    two_form = Forms.FormSpace(2, geom, two_form_space, "u")

    return zero_form, one_form, two_form
end

################################################################################
# Polar B-spline de Rham complex
################################################################################

"""
    create_polar_spline_de_rham_complex(
        num_elements::NTuple{2, Int},
        degrees::NTuple{2, Int},
        regularities::NTuple{2, Int},
        R::Float64;
        refine::Bool=false,
        geom_coeffs_tp::Union{Nothing, Array{Float64,3}}=nothing
    )

Create a polar B-spline de Rham complex.

# Arguments
- `num_elements::NTuple{2, Int}`: the number of elements in each direction.
- `degrees::NTuple{2, Int}`: the degrees of the B-spline spaces.
- `regularities::NTuple{2, Int}`: the regularities of the B-spline spaces.
- `R::Float64`: the radius of the domain.
- `refine::Bool=false`: whether to refine the domain.
- `geom_coeffs_tp::Union{Nothing, Array{Float64,3}}=nothing`: the geometry coefficients.

# Returns
- `::Vector{AbstractFormSpace}`: the 3 form spaces of the complex.
- `::Vector{NTuple{N,SparseMatrixCSC{Float64,Int}} where {N}}`: the global extraction operators.
- `::NTuple{2, Array{Float64,3}}`: the geometry coefficients for the underlying tensor-product B-spline spaces.
"""
function create_polar_spline_de_rham_complex(
    num_elements::NTuple{2, Int},
    degrees::NTuple{2, Int},
    regularities::NTuple{2, Int};
    geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
    R::Float64=1.0,
    two_poles::Bool=false,
    box_sizes::NTuple{2, Float64}=(1.0, 1.0),
    refine::Bool=false,
)
    return create_polar_spline_de_rham_complex(
        num_elements,
        FunctionSpaces.Bernstein.(degrees),
        regularities;
        geom_coeffs_tp=geom_coeffs_tp,
        R=R,
        two_poles=two_poles,
        box_sizes=box_sizes,
        refine=refine,
    )
end

"""
    create_polar_spline_de_rham_complex(
        num_elements::NTuple{2, Int},
        section_spaces::F,
        regularities::NTuple{2, Int},
        R::Float64;
        refine::Bool=false,
        geom_coeffs_tp::Union{Nothing, Array{Float64,3}}=nothing
    ) where {F <: NTuple{2, FunctionSpaces.AbstractCanonicalSpace}}

Create a polar B-spline de Rham complex.

# Arguments
- `num_elements::NTuple{2, Int}`: the number of elements in each direction.
- `section_spaces::F`: the section spaces.
- `regularities::NTuple{2, Int}`: the regularities of the B-spline spaces.
- `R::Float64`: the radius of the domain.
- `refine::Bool=false`: whether to refine the domain.
- `geom_coeffs_tp::Union{Nothing, Array{Float64,3}}=nothing`: the geometry coefficients.

# Returns
- `::Vector{AbstractFormSpace}`: the 3 form spaces of the complex.
- `::Vector{NTuple{N,SparseMatrixCSC{Float64,Int}} where {N}}`: the global extraction operators.
- `::NTuple{2, Array{Float64,3}}`: the geometry coefficients for the underlying tensor-product B-spline spaces.
"""
function create_polar_spline_de_rham_complex(
    num_elements::NTuple{2, Int},
    section_spaces::F,
    regularities::NTuple{2, Int};
    geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
    R::Float64=1.0,
    two_poles::Bool=false,
    box_sizes::NTuple{2, Float64}=(1.0, 1.0),
    refine::Bool=false,
) where {F <: NTuple{2, FunctionSpaces.AbstractCanonicalSpace}}
    form_spaces = Vector{AbstractFormSpace}(undef, 3)

    ##############################
    # Geometry
    ##############################
    P_geom, geom_coeffs_polar = FunctionSpaces.create_polar_geometry_data(
        num_elements,
        section_spaces,
        regularities;
        geom_coeffs_tp=geom_coeffs_tp,
        R=R,
        two_poles=two_poles,
        box_sizes=box_sizes,
    )
    if refine
        P_geom, geom_coeffs_polar, num_elements = FunctionSpaces.refine_geometry_data(
            P_geom, geom_coeffs_polar; two_poles=two_poles
        )
    end
    geom_coeffs_tp = FunctionSpaces.get_degenerate_control_points(P_geom)
    geometry = Geometry.FEGeometry(P_geom, geom_coeffs_polar)

    ##############################
    # 0-Forms
    ##############################
    P⁰ = FunctionSpaces.create_scalar_polar_spline_space(
        num_elements,
        section_spaces,
        regularities;
        geom_coeffs_tp=geom_coeffs_tp,
        R=R,
        two_poles=two_poles,
        zero_at_poles=false,
        box_sizes=box_sizes,
    )
    form_spaces[1] = FormSpace(0, geometry, P⁰, "ω_0")

    ##############################
    # 1-Forms
    ##############################
    P¹ = FunctionSpaces.create_vector_polar_spline_space(
        num_elements,
        section_spaces,
        regularities;
        geom_coeffs_tp=geom_coeffs_tp,
        R=R,
        two_poles=two_poles,
        box_sizes=box_sizes,
    )
    form_spaces[2] = FormSpace(1, geometry, P¹, "ω_1")

    ##############################
    # 2-Forms
    ##############################
    P² = FunctionSpaces.create_scalar_polar_spline_space(
        num_elements,
        section_spaces,
        regularities;
        geom_coeffs_tp=geom_coeffs_tp,
        R=R,
        two_poles=two_poles,
        zero_at_poles=true,
        box_sizes=box_sizes,
    )
    form_spaces[3] = FormSpace(2, geometry, P², "ω_2")

    return form_spaces
end

############################################################################################
#                                   Boundary conditions                                    #
############################################################################################

"""
    set_dirichlet_boundary_conditions(form::AbstractFormSpace, value::Float64)

Creates a dictionary of Dirichlet boundary conditions for a given form space.

# Arguments
- `form::AbstractFormSpace`: The form for which to compute the boundary conditions.
- `value::Float64`: The value of the Dirichlet boundary condition.

# Returns
- `::Dict{Int, Float64}`: The dictionary of Dirichlet boundary conditions.
"""
function set_dirichlet_boundary_conditions(form::AbstractFormSpace, value::Float64)
    return Dict{Int, Float64}(i => value for i in trace_basis_idxs(form))
end

"""
    trace_basis_idxs(
        form::AbstractForm{manifold_dim, form_rank, expression_rank, G}
    ) where {manifold_dim, form_rank, expression_rank, G}

Creates a list of basis function idxs which control the trace of the form on the boundary.

# Arguments
- `form::AbstractForm`: The form for which to compute the boundary conditions.

# Returns
- `Vector{Int}`: The list of basis idxs.
"""
function trace_basis_idxs(
    form::AbstractForm{manifold_dim, form_rank, expression_rank, G}
) where {manifold_dim, form_rank, expression_rank, G}
    if FunctionSpaces.get_num_patches(get_fe_space(form)) > 1
        # This will require topological information to know which interfaces are outer
        # boundaries.
        throw(ArgumentError("trace_basis_idxs not implemented for multipatch geometries"))
    end
    dof_partition = FunctionSpaces.get_dof_partition(get_fe_space(form))
    num_sides = 3^manifold_dim
    basis_idxs = [
        i for j in setdiff(1:num_sides, Int((num_sides + 1) / 2)) for
        i in dof_partition[1][j]
    ]
    return basis_idxs
end

############################################################################################
#                                    Form construction                                     #
############################################################################################
function build_form_field(
    form_space::AbstractFormSpace; label::Union{AbstractString, Nothing}=nothing
)
    if isnothing(label)
        new_label = Forms.get_label(form_space)
    else
        new_label = label
    end

    return FormField(form_space, zeros(get_num_basis(form_space)), new_label)
end

function build_form_field(
    form_space::AbstractFormSpace,
    coeffs::Vector{Float64};
    label::Union{AbstractString, Nothing}=nothing,
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

    if isnothing(label)
        new_label = Forms.get_label(form_space)
    else
        new_label = label
    end

    return FormField(form_space, coeffs, new_label)
end

function build_form_fields(
    form_spaces::FS; labels::Union{L, Nothing}=nothing
) where {
    num_forms,
    FS <: NTuple{num_forms, AbstractFormSpace},
    L <: NTuple{num_forms, AbstractString},
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
            form_spaces[i], coeffs[start_id:(start_id + num_coeffs - 1)]; label=labels[i]
        )
        start_id += num_coeffs

        return ff
    end

    return form_fields
end

################################################################################
# Tensor-product B-spline de Rham complex
################################################################################

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
- `::Tuple{<:AbstractFormSpace{manifold_dim, form_rank}}`: Tuple with the form
    spaces of the complex, for each `form_rank` from `0` to `manifold_dim+1`.
"""
function create_tensor_product_bspline_de_rham_complex(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    section_spaces::NTuple{manifold_dim, F},
    regularities::NTuple{manifold_dim, Int},
) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace}

    # number of dofs on the left and right boundary of the domain
    # n_dofs_left = tuple((1 for _ in 1:manifold_dim)...)
    # n_dofs_right = tuple((1 for _ in 1:manifold_dim)...)

    # store all univariate FEM spaces helper
    fem_spaces = Vector{NTuple{manifold_dim, FunctionSpaces.AbstractFESpace{1, 1}}}(
        undef, 2
    )
    # first, create all univariate FEM spaces corresponding to directional-zero forms
    fem_spaces[1] = FunctionSpaces.create_dim_wise_bspline_spaces(
        starting_points, box_sizes, num_elements, section_spaces, regularities
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
            # Build and store factor spaces of the tensor-product FEM space.
            tp_factor_spaces = ntuple(manifold_dim) do dim
                return fem_spaces[fem_space_idxs[dim]][dim]
            end
            # Build and return the corresponding tensor-product FEM space.
            return FunctionSpaces.TensorProductSpace(tp_factor_spaces)
        end
        if num_form_components == 1
            return FormSpace(k, k_form_fem_spaces[1], "ω_$k")
        else
            return FormSpace(k, FunctionSpaces.DirectSumSpace(k_form_fem_spaces), "ω_$k")
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
		mapping::M,
    ) where {
        manifold_dim,
        F <: FunctionSpaces.AbstractCanonicalSpace,
    	M <: Geometry.AbstractMapping{manifold_dim},
    }

Create a tensor-product B-spline de Rham complex.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: the starting points of the domain.
- `box_sizes::NTuple{manifold_dim, Float64}`: the sizes of the domain.
- `num_elements::NTuple{manifold_dim, Int}`: the number of elements in each direction.
- `section_spaces::NTuple{manifold_dim, F}`: the section spaces.
- `regularities::NTuple{manifold_dim, Int}`: the regularities of the B-spline spaces.
- `mapping::M`: the mapping that applied to be base geometry.

# Returns
- `::Tuple{<:AbstractFormSpace{manifold_dim, form_rank}}`: Tuple with the form
    spaces of the complex, for each `form_rank` from `0` to `manifold_dim+1`.
"""
function create_tensor_product_bspline_de_rham_complex(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    section_spaces::NTuple{manifold_dim, F},
    regularities::NTuple{manifold_dim, Int},
    mapping::M,
) where {
    manifold_dim,
    F <: FunctionSpaces.AbstractCanonicalSpace,
    M <: Geometry.AbstractMapping{manifold_dim},
}

    # number of dofs on the left and right boundary of the domain
    # n_dofs_left = tuple((1 for _ in 1:manifold_dim)...)
    # n_dofs_right = tuple((1 for _ in 1:manifold_dim)...)

    # store all univariate FEM spaces helper
    fem_spaces = Vector{NTuple{manifold_dim, FunctionSpaces.AbstractFESpace{1, 1}}}(
        undef, 2
    )
    # first, create all univariate FEM spaces corresponding to directional-zero forms
    fem_spaces[1] = FunctionSpaces.create_dim_wise_bspline_spaces(
        starting_points, box_sizes, num_elements, section_spaces, regularities
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
            # Build and store factor spaces of the tensor-product FEM space.
            tp_factor_spaces = ntuple(manifold_dim) do dim
                return fem_spaces[fem_space_idxs[dim]][dim]
            end
            # Build and return the corresponding tensor-product FEM space.
            return FunctionSpaces.TensorProductSpace(tp_factor_spaces, mapping)
        end
        if num_form_components == 1
            return FormSpace(k, k_form_fem_spaces[1], "ω_$k")
        else
            return FormSpace(k, FunctionSpaces.DirectSumSpace(k_form_fem_spaces), "ω_$k")
        end
    end

    return form_spaces
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
    c::Float64=0.1,
) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace}
    mapping = Geometry.create_curvilinear_mapping(starting_points, box_sizes, c)

    return create_tensor_product_bspline_de_rham_complex(
        starting_points, box_sizes, num_elements, section_spaces, regularities, mapping
    )
end

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
    c::Float64=0.1,
) where {manifold_dim}
    return create_curvilinear_tensor_product_bspline_de_rham_complex(
        starting_points,
        box_sizes,
        num_elements,
        map(FunctionSpaces.Bernstein, degrees),
        regularities;
        c=c,
    )
end

############################################################################################
#                               Hierarchical de Rham complex                               #
############################################################################################

"""
	create_hierarchical_de_rham_complex(
	    starting_points::NTuple{manifold_dim, Float64},
	    box_sizes::NTuple{manifold_dim, Float64},
	    num_elements::NTuple{manifold_dim, Int},
	    section_spaces::NTuple{manifold_dim, F},
	    regularities::NTuple{manifold_dim, Int},
	    num_subdivisions::NTuple{manifold_dim, Int},
	    ::Type{S},
	    ::Type{B},
	) where {
	    manifold_dim,
	    F <: FunctionSpaces.AbstractCanonicalSpace,
	    S <: FunctionSpaces.SelectionAlgorithm,
	    B <: FunctionSpaces.BasisType,
	}

Construct a hierarchical discrete de Rham complex of finite element spaces over a
tensor-product geometry, equivalent to a Cartesian grid, in `manifold_dim` dimensions.

This routine initializes, for each form degree `k = 0,…,manifold_dim`, a hierarchical
B-spline space of differential `k`‑forms without refinement. All the spaces share the same
[`Geometry.HierarchicalGeometry`](@ref), whose finest level is obtained by subdividing the
elements of the coarsest level by `num_subdivisions`, and is empty until refined with
[`update_hierarchical_de_rham_complex`](@ref).

!!! note
    Only complexes where all form spaces have a single component are supported, which
    currently restricts this routine to `manifold_dim = 1`. Multi-component hierarchical
    spaces are waiting for the multi-component extraction overhaul.

See also [`create_tensor_product_bspline_de_rham_complex`](@ref) and
[`FunctionSpaces.HierarchicalSpace`](@ref).

# Arguments
- `::Type{S}`: The algorithm used to select the active basis functions, such as
    [`FunctionSpaces.SelectionStandard`](@ref) or [`FunctionSpaces.SelectionSimple`](@ref).
- `::Type{B}`: The hierarchical basis representation, such as truncated hierarchical
    B-splines, [`FunctionSpaces.THB`](@ref), or hierarchical B-splines,
    [`FunctionSpaces.HB`](@ref).

# Returns
- A tuple with the `manifold_dim + 1` spaces that form the de Rham complex.
"""
function create_hierarchical_de_rham_complex(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    section_spaces::NTuple{manifold_dim, F},
    regularities::NTuple{manifold_dim, Int},
    num_subdivisions::NTuple{manifold_dim, Int},
    ::Type{S},
    ::Type{B},
) where {
    manifold_dim,
    F <: FunctionSpaces.AbstractCanonicalSpace,
    S <: FunctionSpaces.SelectionAlgorithm,
    B <: FunctionSpaces.BasisType,
}
    # store all univariate FEM spaces helper
    fem_spaces = Vector{NTuple{manifold_dim, FunctionSpaces.AbstractFESpace{1, 1}}}(
        undef, 2
    )
    # first, create all univariate FEM spaces corresponding to directional-zero forms
    fem_spaces[1] = FunctionSpaces.create_dim_wise_bspline_spaces(
        starting_points, box_sizes, num_elements, section_spaces, regularities
    )
    # next, create all univariate FEM spaces corresponding to directional-one forms
    fem_spaces[2] = map(FunctionSpaces.get_derivative_space, fem_spaces[1])

    # Build the tensor-product FEM spaces of all the forms on the coarsest level.
    tp_spaces = ntuple(manifold_dim + 1) do k
        k = k - 1 # Because form ranks range from 0 to manifold_dim.
        # Get k-form basis indices, these also inform the directional degree-deficits.
        k_form_basis_idxs = get_basis_index_combinations(manifold_dim, k)
        if length(k_form_basis_idxs) > 1
            # TODO: Waiting for multi-component extraction overhaul.
            # https://github.com/MantisFEM/Mantis.jl/discussions/421
            return throw(
                ArgumentError(
                    LazyString(
                        "Hierarchical de Rham complexes with multi-component form ",
                        "spaces are not supported yet. Got ",
                        length(k_form_basis_idxs),
                        " components for the ",
                        k,
                        "-forms in ",
                        manifold_dim,
                        " dimensions.",
                    ),
                ),
            )
        end

        # By default, use direction-zero forms...
        fem_space_idxs = ones(Int, manifold_dim)
        # ...unless the basis index is present in the k-form basis indices.
        fem_space_idxs[first(k_form_basis_idxs)] .= 2
        # Build the tensor-product FEM space from its factor spaces.
        tp_factor_spaces = ntuple(manifold_dim) do dim
            return fem_spaces[fem_space_idxs[dim]][dim]
        end

        return FunctionSpaces.TensorProductSpace(tp_factor_spaces)
    end

    # The spaces of all forms share the same hierarchical geometry. Its coarsest level is
    # fully active, while its finest level is empty until the complex is refined.
    basis_scalings = map(
        space -> FunctionSpaces.scaling_uniform(space, num_subdivisions), tp_spaces
    )
    geometry_scaling = Geometry.scaling_uniform(
        FunctionSpaces.get_parametric_geometry(first(tp_spaces)),
        FunctionSpaces.get_parametric_geometry(get_child(first(basis_scalings))),
        num_subdivisions,
    )
    active_elements = ActiveInfo([collect(1:prod(num_elements)), Int[]])
    geometry = Geometry.HierarchicalGeometry(active_elements, (geometry_scaling,))

    form_spaces = ntuple(manifold_dim + 1) do k
        hierarchical_space = FunctionSpaces.HierarchicalSpace(
            geometry, (basis_scalings[k],), S, B
        )

        return FormSpace(k - 1, hierarchical_space, "ω_$(k - 1)")
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
    ::Type{S},
    ::Type{B},
) where {
    manifold_dim, S <: FunctionSpaces.SelectionAlgorithm, B <: FunctionSpaces.BasisType
}
    return create_hierarchical_de_rham_complex(
        starting_points,
        box_sizes,
        num_elements,
        map(FunctionSpaces.Bernstein, degrees),
        regularities,
        num_subdivisions,
        S,
        B,
    )
end

"""
	update_hierarchical_de_rham_complex(
		complex::C,
		elements::Vector{Int};
		num_subdivisions::NTuple{manifold_dim, Int}=ntuple(_ -> 2, manifold_dim),
	) where {manifold_dim, num_forms, C <: NTuple{num_forms, AbstractFormSpace{manifold_dim}}}

Returns a refined hierarchical de Rham complex, based on the given `complex` and the marked
`elements`.

The marked elements are refined in the hierarchical geometry shared by all the spaces of the
complex, and then all the spaces are updated to the refined geometry. If an element on the
finest level is marked, a new level is created by subdividing the elements of the finest
level by `num_subdivisions`.

See also [`create_hierarchical_de_rham_complex`](@ref) and
[`FunctionSpaces.refine`](@ref).

# Arguments
- `complex::C`: The hierarchical B-spline de Rham complex, for example created with
    [`create_hierarchical_de_rham_complex`](@ref).
- `elements::Vector{Int}`: The marked elements, in hierarchical indexing, for example
    obtained with a [`FunctionSpaces.MarkingStrategy`](@ref).

# Keyword Arguments
- `num_subdivisions::NTuple{manifold_dim, Int}`: The number of subdivisions per dimension
    used if a new level is created.

# Returns
- `new_complex<:NTuple{num_forms, AbstractFormSpace}`:A tuple with the `manifold_dim + 1`
	refined spaces that form the de Rham complex.
"""
function update_hierarchical_de_rham_complex(
    complex::C,
    elements::Vector{Int};
    num_subdivisions::NTuple{manifold_dim, Int}=ntuple(_ -> 2, manifold_dim),
) where {manifold_dim, num_forms, C <: NTuple{num_forms, AbstractFormSpace{manifold_dim}}}
    spaces = map(get_fe_space, complex)
    for space in spaces
        if !(space isa FunctionSpaces.HierarchicalSpace)
            # TODO: Waiting for multi-component extraction overhaul.
            # https://github.com/MantisFEM/Mantis.jl/discussions/421
            return throw(
                ArgumentError(
                    LazyString(
                        "All the spaces of the complex should be single-component ",
                        "hierarchical spaces. Got ",
                        Base.typename(typeof(space)).wrapper,
                    ),
                ),
            )
        end
    end

    geometry = FunctionSpaces.get_parametric_geometry(first(spaces))
    if any(space -> !(FunctionSpaces.get_parametric_geometry(space) === geometry), spaces)
        return throw(
            ArgumentError(
                "All the spaces of the complex should share the same hierarchical geometry."
            ),
        )
    end

    # The marked elements are refined level by level, so we convert them to level indexing.
    elements_per_level = [Int[] for _ in 1:Geometry.get_num_levels(geometry)]
    for element_id in unique(elements)
        level, level_id = Geometry.convert_to_level_and_level_id(geometry, element_id)
        push!(elements_per_level[level], level_id)
    end

    # The scalings are only used if a new level is created. The new level of the geometry
    # has to be the parametric geometry of the new level of the spaces.
    basis_scalings = map(spaces) do space
        last_space = get_child(last(get_scalings(FunctionSpaces.get_basis(space))))

        return FunctionSpaces.scaling_uniform(last_space, num_subdivisions)
    end
    last_geometry = get_child(last(get_scalings(Geometry.get_hierarchy(geometry))))
    geometry_scaling = Geometry.scaling_uniform(
        last_geometry,
        FunctionSpaces.get_parametric_geometry(get_child(first(basis_scalings))),
        num_subdivisions,
    )
    for (level, level_elements) in enumerate(elements_per_level)
        if !isempty(level_elements)
            geometry = Geometry.refine(geometry, level, level_elements, geometry_scaling)
        end
    end

    new_complex = ntuple(num_forms) do k
        space = FunctionSpaces.update(spaces[k], geometry, geometry, basis_scalings[k])

        return FormSpace(get_form_rank(complex[k]), space, get_label(complex[k]))
    end

    return new_complex
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
    R::Float64=1.0,
    two_poles::Bool=false,
    box_sizes::NTuple{2, Float64}=(1.0, 1.0),
    # refine::Bool=false,
) where {F <: NTuple{2, FunctionSpaces.AbstractCanonicalSpace}}
    form_spaces = Vector{AbstractFormSpace}(undef, 3)

    ##############################
    # Geometry
    ##############################
    geometry, geom_coeffs_tp = FunctionSpaces.create_polar_geometry_data(
        num_elements, section_spaces, regularities; R=R, box_sizes=box_sizes
    )
    # if refine
    #     P_geom, geom_coeffs_tp, num_elements = FunctionSpaces.refine_geometry_data(
    #         P_geom, geom_coeffs_polar; two_poles=two_pole
    #     )
    # end

    ##############################
    # 0-Forms
    ##############################
    P⁰ = FunctionSpaces.create_scalar_polar_spline_space(
        num_elements,
        section_spaces,
        regularities,
        geometry;
        geom_coeffs_tp=geom_coeffs_tp,
        two_poles=two_poles,
        zero_at_poles=false,
        box_sizes=box_sizes,
    )
    form_spaces[1] = FormSpace(0, P⁰, "ω_0")

    ##############################
    # 1-Forms
    ##############################
    P¹ = FunctionSpaces.create_vector_polar_spline_space(
        num_elements,
        section_spaces,
        regularities,
        geometry;
        geom_coeffs_tp=geom_coeffs_tp,
        two_poles=two_poles,
        box_sizes=box_sizes,
    )
    form_spaces[2] = FormSpace(1, P¹, "ω_1")

    ##############################
    # 2-Forms
    ##############################
    P² = FunctionSpaces.create_scalar_polar_spline_space(
        num_elements,
        section_spaces,
        regularities,
        geometry;
        geom_coeffs_tp=geom_coeffs_tp,
        two_poles=two_poles,
        zero_at_poles=true,
        box_sizes=box_sizes,
    )
    form_spaces[3] = FormSpace(2, P², "ω_2")

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
        form::AbstractForm{manifold_dim, form_rank, expression_rank}
    ) where {manifold_dim, form_rank, expression_rank}

Creates a list of basis function idxs which control the trace of the form on the boundary.

# Arguments
- `form::AbstractForm`: The form for which to compute the boundary conditions.

# Returns
- `Vector{Int}`: The list of basis idxs.
"""
function trace_basis_idxs(
    form::AbstractForm{manifold_dim, form_rank, expression_rank}
) where {manifold_dim, form_rank, expression_rank}
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

############################################################################################
#                                    Component mixing                                      #
############################################################################################

# The components of a form evaluation can each have their own set of (global) basis indices,
# see [`evaluate`](@ref). Operators that combine several components into a single output
# component (e.g. `d`, `★`, `∧`) evaluate that output component on the union of the indices
# of the combined components, and add the contribution of each component to the matching
# columns. The helpers below implement this.

"""
    _field_indices(num_components::Int)

Returns the (dummy) indices of an expression with expression rank 0 that has
`num_components` components.

# Returns
- `::Vector{Vector{Vector{Int}}}`: `[[1]]` for each component.
"""
_field_indices(num_components::Int) = [[[1]] for _ in 1:num_components]

"""
    _merge_component_indices(
        indices::Vector{Vector{Vector{Int}}},
        components,
        ::Val{expression_rank},
    ) where {expression_rank}

Merges the indices of the given `components` of a form evaluation, per expression.

# Arguments
- `indices::Vector{Vector{Vector{Int}}}`: The indices of a form evaluation, where
    `indices[c][e]` are the global basis indices of expression `e` of component `c`. See
    [`evaluate`](@ref).
- `components`: Iterable with the ids of the components to merge.
- `::Val{expression_rank}`: The number of expressions to merge. Should be constructed from a
    type parameter, so that it is known at compile time.

# Returns
- `merged_indices::NTuple{expression_rank, Vector{Int}}`: For each expression, the sorted
    union of the indices of the merged components.
- `column_maps::Vector{NTuple{expression_rank, Vector{Int}}}`: For each component of
    `indices` (not only the merged ones), and for each expression, the position in
    `merged_indices` of each of the component's indices. The maps of components that are not
    merged are empty.
"""
function _merge_component_indices(
    indices::Vector{Vector{Vector{Int}}}, components, ::Val{expression_rank}
) where {expression_rank}
    # Sorting gives a canonical order, so equal sets give equal vectors, and allows the
    # column maps to be found by binary search.
    # This has the benefit of avoiding `find` which would return `Union{Nothing, Int}`,
    # allowing us to use `searchsortedfirst` which always returns an `Int`.
    merged_indices = ntuple(Val(expression_rank)) do expression_id
        merged = Int[]
        for component_id in components
            append!(merged, indices[component_id][expression_id])
        end

        return sort!(unique!(merged))
    end

    column_maps = [ntuple(_ -> Int[], Val(expression_rank)) for _ in eachindex(indices)]
    for component_id in components
        column_maps[component_id] = ntuple(Val(expression_rank)) do expression_id
            return [
                searchsortedfirst(merged_indices[expression_id], index) for
                index in indices[component_id][expression_id]
            ]
        end
    end

    return merged_indices, column_maps
end

"""
    _allocate_merged(
        num_points::Int, merged_indices::NTuple{expression_rank, Vector{Int}}
    ) where {expression_rank}

Allocates a zero array to store the evaluation of a component with the given merged indices.

# Returns
- `::Array{Float64, 1 + expression_rank}`: Zero array of size
    `(num_points, length.(merged_indices)...)`.
"""
function _allocate_merged(
    num_points::Int, merged_indices::NTuple{expression_rank, Vector{Int}}
) where {expression_rank}
    return zeros(Float64, num_points, map(length, merged_indices)...)
end

"""
    _merged_to_indices(
        merged_indices::NTuple{expression_rank, Vector{Int}}
    ) where {expression_rank}

Converts the merged indices of a component to the per-expression indices format of
[`evaluate`](@ref). Expressions of rank 0 get the dummy indices `[[1]]`.
"""
function _merged_to_indices(
    merged_indices::NTuple{expression_rank, Vector{Int}}
) where {expression_rank}
    if expression_rank == 0
        return [[1]]
    end

    return collect(merged_indices)
end

_get_weight(weight::Number, ::Int) = weight
_get_weight(weights::AbstractVector, point_id::Int) = weights[point_id]

"""
    _add_mapped!(
        out::AbstractArray{Float64},
        values::AbstractArray{Float64},
        column_maps::NTuple{expression_rank, Vector{Int}},
        weights,
    ) where {expression_rank}

Adds the evaluation `values` of a single component, scaled per point by `weights`, to the
columns of `out` given by `column_maps`, i.e.
`out[p, column_maps[1][j₁], …] += weights[p] * values[p, j₁, …]`.

# Arguments
- `out::AbstractArray{Float64}`: Array of the merged evaluation. Modified in-place.
- `values::AbstractArray{Float64}`: Evaluation of the component to add.
- `column_maps::NTuple{expression_rank, Vector{Int}}`: Column map of the component per
    expression, see [`_merge_component_indices`](@ref).
- `weights`: Either a number, or a vector with one weight per evaluation point.

# Returns
- `out::AbstractArray{Float64}`.
"""
function _add_mapped!(
    out::AbstractArray{Float64},
    values::AbstractArray{Float64},
    column_maps::NTuple{expression_rank, Vector{Int}},
    weights,
) where {expression_rank}
    basis_ids = CartesianIndices(ntuple(e -> length(column_maps[e]), Val(expression_rank)))
    for basis_id in basis_ids
        out_id = ntuple(e -> column_maps[e][basis_id[e]], Val(expression_rank))
        for point_id in axes(values, 1)
            out[point_id, out_id...] +=
                _get_weight(weights, point_id) * values[point_id, Tuple(basis_id)...]
        end
    end

    return out
end

"""
    _add_mapped_product!(
        out::AbstractArray{Float64},
        sign::Int,
        values_1::AbstractArray{Float64},
        column_maps_1::NTuple{expression_rank_1, Vector{Int}},
        values_2::AbstractArray{Float64},
        column_maps_2::NTuple{expression_rank_2, Vector{Int}},
    ) where {expression_rank_1, expression_rank_2}

Adds the (outer) product of the evaluations of two components to the columns of `out` given
by the column maps, i.e.
`out[p, map₁(j₁)…, map₂(j₂)…] += sign * values_1[p, j₁…] * values_2[p, j₂…]`.

# Returns
- `out::AbstractArray{Float64}`.
"""
function _add_mapped_product!(
    out::AbstractArray{Float64},
    sign::Int,
    values_1::AbstractArray{Float64},
    column_maps_1::NTuple{expression_rank_1, Vector{Int}},
    values_2::AbstractArray{Float64},
    column_maps_2::NTuple{expression_rank_2, Vector{Int}},
) where {expression_rank_1, expression_rank_2}
    basis_ids_1 = CartesianIndices(
        ntuple(e -> length(column_maps_1[e]), Val(expression_rank_1))
    )
    basis_ids_2 = CartesianIndices(
        ntuple(e -> length(column_maps_2[e]), Val(expression_rank_2))
    )
    for basis_id_2 in basis_ids_2
        out_id_2 = ntuple(e -> column_maps_2[e][basis_id_2[e]], Val(expression_rank_2))
        for basis_id_1 in basis_ids_1
            out_id_1 = ntuple(e -> column_maps_1[e][basis_id_1[e]], Val(expression_rank_1))
            for point_id in axes(out, 1)
                out[point_id, out_id_1..., out_id_2...] +=
                    sign *
                    values_1[point_id, Tuple(basis_id_1)...] *
                    values_2[point_id, Tuple(basis_id_2)...]
            end
        end
    end

    return out
end

"""
    _mix_components(
        values::Vector{<:AbstractArray{Float64}},
        indices::Vector{Vector{Vector{Int}}},
        components,
        weights,
        ::Val{expression_rank},
    ) where {expression_rank}

Computes the linear combination `∑ᵢ weights[i] * values[components[i]]` of components of a
form evaluation, on the union of the indices of these components.

# Arguments
- `values::Vector{<:AbstractArray{Float64}}`: The evaluation per component.
- `indices::Vector{Vector{Vector{Int}}}`: The indices per component, see [`evaluate`](@ref).
- `components`: Indexable collection with the ids of the components to combine.
- `weights`: Indexable collection with the weight of each component in `components`. Each
    weight is either a number or a vector with a weight per evaluation point.
- `::Val{expression_rank}`: The expression rank, constructed from a type parameter.

# Returns
- `::Array{Float64, 1 + expression_rank}`: The combined evaluation.
- `::Vector{Vector{Int}}`: The indices of the combined evaluation, per expression.
"""
function _mix_components(
    values::Vector{<:AbstractArray{Float64}},
    indices::Vector{Vector{Vector{Int}}},
    components,
    weights,
    ::Val{expression_rank},
) where {expression_rank}
    merged_indices, column_maps = _merge_component_indices(
        indices, components, Val(expression_rank)
    )
    mixed_values = _allocate_merged(size(values[first(components)], 1), merged_indices)
    for i in eachindex(components, weights)
        component_id = components[i]
        _add_mapped!(
            mixed_values, values[component_id], column_maps[component_id], weights[i]
        )
    end

    return mixed_values, _merged_to_indices(merged_indices)
end

"""
    _expand_to_merged(
        values_1::AbstractArray{Float64},
        indices_1::Vector{Vector{Int}},
        values_2::AbstractArray{Float64},
        indices_2::Vector{Vector{Int}},
        ::Val{expression_rank},
    ) where {expression_rank}

Expands two evaluations with (possibly) different indices to the union of their indices, so
that they can be combined entry-wise. Entries of an evaluation outside its own indices are
zero.

# Arguments
- `values_1::AbstractArray{Float64}`, `values_2::AbstractArray{Float64}`: The evaluations.
- `indices_1::Vector{Vector{Int}}`, `indices_2::Vector{Vector{Int}}`: The indices of each
    evaluation, per expression.
- `::Val{expression_rank}`: The number of expressions, constructed from a type parameter.

# Returns
- `::Array{Float64, 1 + expression_rank}`: The expanded `values_1`.
- `::Array{Float64, 1 + expression_rank}`: The expanded `values_2`.
- `::Vector{Vector{Int}}`: The merged indices, per expression.
"""
function _expand_to_merged(
    values_1::AbstractArray{Float64},
    indices_1::Vector{Vector{Int}},
    values_2::AbstractArray{Float64},
    indices_2::Vector{Vector{Int}},
    ::Val{expression_rank},
) where {expression_rank}
    merged_indices, column_maps = _merge_component_indices(
        [indices_1, indices_2], (1, 2), Val(expression_rank)
    )
    num_points = size(values_1, 1)
    expanded_1 = _allocate_merged(num_points, merged_indices)
    _add_mapped!(expanded_1, values_1, column_maps[1], 1.0)
    expanded_2 = _allocate_merged(num_points, merged_indices)
    _add_mapped!(expanded_2, values_2, column_maps[2], 1.0)

    return expanded_1, expanded_2, _merged_to_indices(merged_indices)
end

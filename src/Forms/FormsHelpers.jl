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

    # Build the FEM spaces of all the forms on the coarsest level.
    level_spaces = ntuple(manifold_dim + 1) do k
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
            # Build the tensor-product FEM space from its factor spaces.
            tp_factor_spaces = ntuple(manifold_dim) do dim
                return fem_spaces[fem_space_idxs[dim]][dim]
            end

            return FunctionSpaces.TensorProductSpace(tp_factor_spaces)
        end
        if num_form_components == 1
            return k_form_fem_spaces[1]
        end

        return FunctionSpaces.DirectSumSpace(k_form_fem_spaces)
    end

    # The spaces of all forms share the same hierarchical geometry. Its coarsest level is
    # fully active, while its finest level is empty until the complex is refined.
    basis_scalings = map(
        space -> FunctionSpaces.scaling_uniform(space, num_subdivisions), level_spaces
    )
    geometry_scaling = Geometry.scaling_uniform(
        FunctionSpaces.get_parametric_geometry(first(level_spaces)),
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
            return throw(
                ArgumentError(
                    LazyString(
                        "All the spaces of the complex should be hierarchical spaces. ",
                        "Got ",
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

# Examples
```jldoctest
using Mantis

Mantis.Forms._field_indices(2)

# output

2-element Vector{Vector{Vector{Int64}}}:
 [[1]]
 [[1]]
```
"""
_field_indices(num_components::Int) = [[[1]] for _ in 1:num_components]

"""
    _combine_indices(
        indices::Vector{Vector{Vector{Int}}},
        components::AbstractVector{Int},
        ::Val{expression_rank},
    ) where {expression_rank}

Combines the indices of the given `components` of a form evaluation, per expression.

# Arguments
- `indices::Vector{Vector{Vector{Int}}}`: The indices of a form evaluation, where
    `indices[c][e]` are the global basis indices of expression `e` of component `c`. See
    [`evaluate`](@ref).
- `components::AbstractVector{Int}`: The ids of the components to combine.
- `::Val{expression_rank}`: The number of expressions to combine. Should be constructed from
    a type parameter, so that it is known at compile time.

# Returns
- `combined_indices::NTuple{expression_rank, Vector{Int}}`: For each expression, the sorted
    union of the indices of the combined components.
- `column_maps::Vector{NTuple{expression_rank, Vector{Int}}}`: For each component of
    `indices` (not only the combined ones), and for each expression, the position in
    `combined_indices` of each of the component's indices. The maps of components that are
    not combined are empty.

# Examples
Two components with expression rank 1, sharing the basis function `5`:
```jldoctest
using Mantis

indices = [[[3, 5, 7]], [[5, 8]]]
Mantis.Forms._combine_indices(indices, [1, 2], Val(1))

# output

(([3, 5, 7, 8],), [([1, 2, 3],), ([2, 4],)])
```

Only combining the second component leaves the map of the first one empty:
```jldoctest
using Mantis

indices = [[[3, 5, 7]], [[5, 8]]]
Mantis.Forms._combine_indices(indices, [2], Val(1))

# output

(([5, 8],), [(Int64[],), ([1, 2],)])
```
"""
function _combine_indices(
    indices::Vector{Vector{Vector{Int}}},
    components::AbstractVector{Int},
    ::Val{expression_rank},
) where {expression_rank}
    # Sorting gives a canonical order, so equal sets give equal vectors, and allows the
    # column maps to be found by binary search (which is what `searchsortedfirst` usually
    # does).
    # This has the benefit of avoiding `find` which would return `Union{Nothing, Int}`,
    # allowing us to use `searchsortedfirst` which always returns an `Int`.
    combined_indices = ntuple(Val(expression_rank)) do expression_id
        expression_indices = Int[]
        for component_id in components
            append!(expression_indices, indices[component_id][expression_id])
        end

        return sort!(unique!(expression_indices))
    end

    column_maps = [ntuple(_ -> Int[], Val(expression_rank)) for _ in eachindex(indices)]
    for component_id in components
        column_maps[component_id] = ntuple(Val(expression_rank)) do expression_id
            return [
                searchsortedfirst(combined_indices[expression_id], index) for
                index in indices[component_id][expression_id]
            ]
        end
    end

    return combined_indices, column_maps
end

"""
    _allocate_indices(
        num_points::Int, indices::NTuple{expression_rank, Vector{Int}}; T=Float64
    ) where {expression_rank}

Allocates a zero array to store the evaluation of a component with the given indices.

# Returns
- `::Array{T, 1 + expression_rank}`: Zero array of size
    `(num_points, length.(indices)...)`.

# Examples
```jldoctest
using Mantis

Mantis.Forms._allocate_indices(2, ([3, 5, 7, 8],))

# output

2×4 Matrix{Float64}:
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
```
"""
function _allocate_indices(
    num_points::Int, indices::NTuple{expression_rank, Vector{Int}}; T=Float64
) where {expression_rank}
    return zeros(T, num_points, map(length, indices)...)
end

"""
    _tuple_to_vec_indices(
        combined_indices::NTuple{expression_rank, Vector{Int}}
    ) where {expression_rank}

Converts the combined indices of a component to the per-expression indices format of
[`evaluate`](@ref). Expressions of rank 0 get the dummy indices `[[1]]`.

# Examples
```jldoctest
using Mantis

Mantis.Forms._tuple_to_vec_indices(([3, 5, 7, 8],))

# output

1-element Vector{Vector{Int64}}:
 [3, 5, 7, 8]
```

```jldoctest
using Mantis

Mantis.Forms._tuple_to_vec_indices(())

# output

1-element Vector{Vector{Int64}}:
 [1]
```
"""
function _tuple_to_vec_indices(
    combined_indices::NTuple{expression_rank, Vector{Int}}
) where {expression_rank}
    if iszero(expression_rank)
        return [[1]]
    end

    return collect(combined_indices)
end

_get_weight(weight::Number, ::Int) = weight
_get_weight(weights::AbstractVector, point_id::Int) = weights[point_id]

"""
    _add_columns!(
        out::AbstractArray{T},
        values::AbstractArray{T},
        column_maps::NTuple{expression_rank, Vector{Int}},
        weights,
    ) where {T, expression_rank}

Adds the evaluation `values` of a single component, scaled per point by `weights`, to the
columns of `out` given by `column_maps`, i.e.
`out[p, column_maps[1][j₁], …] += weights[p] * values[p, j₁, …]`.

# Arguments
- `out::AbstractArray{T}`: Array of the combined evaluation. Modified in-place.
- `values::AbstractArray{T}`: Evaluation of the component to add.
- `column_maps::NTuple{expression_rank, Vector{Int}}`: Column map of the component per
    expression, see [`_combine_indices`](@ref).
- `weights`: Either a number, or a vector with one weight per evaluation point.

# Returns
- `out::AbstractArray{T}`.

# Examples
Computes `α₁ - α₂` on the combined indices `[3, 5, 7, 8]`, where `α₁` has the indices
`[3, 5, 7]` and `α₂` has the indices `[5, 8]` (see [`_combine_indices`](@ref)):
```jldoctest
using Mantis

out = zeros(1, 4)
Mantis.Forms._add_columns!(out, [1.0 2.0 3.0], ([1, 2, 3],), 1.0)
Mantis.Forms._add_columns!(out, [10.0 20.0], ([2, 4],), -1.0)

# output

1×4 Matrix{Float64}:
 1.0  -8.0  3.0  -20.0
```
"""
function _add_columns!(
    out::AbstractArray{T},
    values::AbstractArray{T},
    column_maps::NTuple{expression_rank, Vector{Int}},
    weights,
) where {T, expression_rank}
    local_basis = map(e -> 1:length(e), column_maps)
    for (point, O) in enumerate(eachslice(out; dims=1))
        w = _get_weight(weights, point)
        view(O, column_maps...) .+= w .* view(values, point, local_basis...)
    end

    return out
end

"""
    _add_outer_product!(
        out::AbstractArray{T},
        values_1::AbstractArray{T},
        column_maps_1::NTuple{expression_rank_1, Vector{Int}},
        values_2::AbstractArray{T},
        column_maps_2::NTuple{expression_rank_2, Vector{Int}},
        weights,
    ) where {T, expression_rank_1, expression_rank_2}

Adds the (outer) product of the evaluations of two components, scaled per point by
`weights`, to the columns of `out` given by the column maps, i.e.
`out[p, map₁(j₁)…, map₂(j₂)…] += weights[p] * values_1[p, j₁…] * values_2[p, j₂…]`.

# Arguments
- `out::AbstractArray{T}`: Array of the combined evaluation. Modified in-place.
- `values_1::AbstractArray{T}`, `values_2::AbstractArray{T}`: Evaluations of the components
    to multiply.
- `column_maps_1::NTuple{expression_rank_1, Vector{Int}}`,
    `column_maps_2::NTuple{expression_rank_2, Vector{Int}}`: Column maps of each component per
    expression, see [`_combine_indices`](@ref).
- `weights`: Either a number, or a vector with one weight per evaluation point.

# Returns
- `out::AbstractArray{T}`.

# Examples
Computes `-α ⊗ β`, where the two basis functions of `α` are mapped to the columns `[1, 2]`
of the first expression, and those of `β` to the columns `[3, 1]` of the second
expression:
```jldoctest
using Mantis

out = zeros(1, 2, 3)
Mantis.Forms._add_outer_product!(out, [1.0 2.0], ([1, 2],), [10.0 20.0], ([3, 1],), -1.0)

# output

1×2×3 Array{Float64, 3}:
[:, :, 1] =
 -20.0  -40.0

[:, :, 2] =
 0.0  0.0

[:, :, 3] =
 -10.0  -20.0
```
"""
function _add_outer_product!(
    out::AbstractArray{T},
    values_1::AbstractArray{T},
    column_maps_1::NTuple{expression_rank_1, Vector{Int}},
    values_2::AbstractArray{T},
    column_maps_2::NTuple{expression_rank_2, Vector{Int}},
    weights,
) where {T, expression_rank_1, expression_rank_2}
    local_basis_1 = CartesianIndices(map(e -> 1:length(e), column_maps_1))
    local_basis_2 = CartesianIndices(map(e -> 1:length(e), column_maps_2))
    for b2 in local_basis_2
        c2 = ntuple(e -> column_maps_2[e][b2[e]], expression_rank_2)
        for b1 in local_basis_1
            c1 = ntuple(e -> column_maps_1[e][b1[e]], expression_rank_1)
            for (point, O) in enumerate(eachslice(out; dims=1))
                w = _get_weight(weights, point)
                O[c1..., c2...] +=
                    w * values_1[point, Tuple(b1)...] * values_2[point, Tuple(b2)...]
            end
        end
    end

    return out
end

############################################################################################
#                                    Form construction                                     #
############################################################################################
function build_form_field(
    form_space::AbstractFormSpace; label::Union{String, Nothing}=nothing
)
    if isnothing(label)
        return FormField(form_space, Forms.get_label(form_space))
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
            # Build and store constituent spaces of the tensor-product FEM space.
            tp_consituent_spaces = ntuple(manifold_dim) do dim
                return fem_spaces[fem_space_idxs[dim]][dim]
            end
            # Build and return the corresponding tensor-product FEM space.
            return FunctionSpaces.TensorProductSpace(tp_consituent_spaces)
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
            # Build and store constituent spaces of the tensor-product FEM space.
            tp_consituent_spaces = ntuple(manifold_dim) do dim
                return fem_spaces[fem_space_idxs[dim]][dim]
            end
            # Build and return the corresponding tensor-product FEM space.
            return FunctionSpaces.TensorProductSpace(tp_consituent_spaces, mapping)
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
	    truncate::Bool,
	    simplified::Bool,
	    geometry::G,
	) where {
	    manifold_dim,
	    F <: FunctionSpaces.AbstractCanonicalSpace,
	    G <: Geometry.AbstractGeometry{manifold_dim},
	}

Construct a hierarchical discrete de Rham complex of finite element spaces over a
tensor-product geometry, equivalent to a Cartesian grid, in `manifold_dim` dimensions.

This routine initializes, for each form degree `k = 0,…,manifold_dim`, a hierarchical
B-spline space of differential `k`‑forms without refinement. 

See also [`create_tensor_product_bspline_de_rham_complex`](@ref) and
[`FunctionSpaces.HierarchicalFiniteElementSpace`](@ref).

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
    truncate::Bool,
    simplified::Bool,
) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace}
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
                tp_space, num_subdivisions, truncate, simplified
            )

            return hierarchical_space
        end

        if num_form_components == 1
            return FormSpace(k, k_form_fem_spaces[1], "ω_$k")
        else
            return FormSpace(k, FunctionSpaces.DirectSumSpace(k_form_fem_spaces), "ω_$k")
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
    return create_hierarchical_de_rham_complex(
        starting_points,
        box_sizes,
        num_elements,
        map(FunctionSpaces.Bernstein, degrees),
        regularities,
        num_subdivisions,
        truncate,
        simplified,
    )
end

"""
	update_hierarchical_de_rham_complex(
		complex::C, data
	) where {num_forms, C <: NTuple{num_forms, AbstractFormSpace}}

Returns a refined hierarchical de Rham complex, based on the given `complex` and refinement
`data`. The input `data` should have a dedicated method in
`FunctionSpaces.refine_space(space, data)`.

See also [`FunctionSpaces.refine_space`](@ref).

# Arguments
- `complex::C`: The hierarchical B-spline de Rham complex.
- `data`: The information used for refinement. Examples include domains denoting active
    elements, of type [`Hierarchy.ActiveInfo`](@ref), or elements marked for refinement, of
    type `Vector{Vector{Int}}`.

# Returns
- `new_complex<:NTuple{num_forms, AbstractFormSpace}`:A tuple with the `manifold_dim + 1`
	refined spaces that form the de Rham complex.
"""
function update_hierarchical_de_rham_complex(
    complex::C, data
) where {num_forms, C <: NTuple{num_forms, AbstractFormSpace}}
    new_complex = ntuple(num_forms) do k
        num_components = FunctionSpaces.get_num_components(get_fe_space(complex[k]))
        if num_components == 1
            new_space = FunctionSpaces.refine_space(get_fe_space(complex[k]), data)
        else
            comp_spaces = FunctionSpaces.get_component_spaces(get_fe_space(complex[k]))
            new_space = FunctionSpaces.DirectSumSpace(
                ntuple(num_components) do c
                    FunctionSpaces.refine_space(comp_spaces[c], data)
                end,
            )
        end

        return FormSpace(k - 1, new_space, get_label(complex[k]))
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

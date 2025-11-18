################################################################################
# Univariate and tensor-product B-spline space creation functions
################################################################################

"""
    create_bspline_space(
        starting_point::Float64,
        box_size::Float64,
        num_elements::Int,
        section_space::F,
        regularity::Int;
        n_dofs_left::Int=1,
        n_dofs_right::Int=1,
    ) where {F <: AbstractCanonicalSpace}

Create a 1D B-spline space based on the provided starting point, box size, number of
elements, section space, and regularity. It constructs a uniform distribution of
breakpoints and uses them to define a `Mesh.Patch1D` object. The function then creates a
regularity vector, with the first and last entries set to -1 to ensure an open knot vector,
and returns the corresponding `BSplineSpace`. Optional arguments include the number of
degrees of freedom on the left and right boundaries.

# Arguments
- `starting_point::Float64`: The coordinate of the starting point of the B-spline space.
- `box_size::Float64`: The size of the space along the single dimension.
- `num_elements::Int`: The number of elements along the dimension.
- `section_space::F`: The section space to use for the B-spline basis functions.
- `regularity::Int`: The regularity (smoothness) between consecutive B-spline basis
    functions, typically between 0 and `degree - 1`.
- `n_dofs_left::Int`: The number of degrees of freedom on the left boundary.
- `n_dofs_right::Int`: The number of degrees of freedom on the right boundary.

# Returns
- `::BSplineSpace`: The generated B-spline space.

# Examples
```julia
# Create a 1D B-spline space
start_1d = 0.0
size_1d = 10.0
elements_1d = 100
degree = 3
regularity = 2
section_space = Bernstein(degree)
bspline_space_1d = create_bspline_space(start_1d, size_1d, elements_1d, section_space, regularity)
```
"""
function create_bspline_space(
    starting_point::Float64,
    box_size::Float64,
    num_elements::Int,
    section_space::F,
    regularity::Int;
    n_dofs_left::Int=1,
    n_dofs_right::Int=1,
) where {F <: AbstractCanonicalSpace}
    if box_size > 0.0
        nothing
    else
        throw(ArgumentError("The argument 'box_size' must be greater than 0."))
    end

    breakpoints = collect(
        LinRange(starting_point, starting_point + box_size, num_elements + 1)
    )
    patch = Mesh.Patch1D(breakpoints)

    regularity_vector = fill(regularity, (num_elements + 1,))
    regularity_vector[1] = regularity_vector[end] = -1 # Open knot vector

    return BSplineSpace(patch, section_space, regularity_vector, n_dofs_left, n_dofs_right)
end

"""
    create_bspline_space(
        starting_point::Float64,
        box_size::Float64,
        num_elements::Int,
        degree::Int,
        regularity::Int;
        n_dofs_left::Int=1,
        n_dofs_right::Int=1,
    )

Create a piecewise-polynomial 1D B-spline space based on the provided starting point, box
size, number of elements, polynomial degree, and regularity. It constructs a uniform
distribution of breakpoints and uses them to define a `Mesh.Patch1D` object. The function
then creates a regularity vector, with the first and last entries set to -1 to ensure an
open knot vector, and returns the corresponding `BSplineSpace`. Optional arguments include
the number of degrees of freedom on the left and right boundaries.

# Arguments
- `starting_point::Float64`: The coordinate of the starting point of the B-spline space.
- `box_size::Float64`: The size of the space along the single dimension.
- `num_elements::Int`: The number of elements along the dimension.
- `degree::Int`: The polynomial degree on each element.
- `regularity::Int`: The regularity (smoothness) between consecutive B-spline basis
    functions, typically between 0 and `degree - 1`.
- `n_dofs_left::Int`: The number of degrees of freedom on the left boundary.
- `n_dofs_right::Int`: The number of degrees of freedom on the right boundary.

# Returns
- `::BSplineSpace`: The generated B-spline space.
"""
function create_bspline_space(
    starting_point::Float64,
    box_size::Float64,
    num_elements::Int,
    degree::Int,
    regularity::Int;
    n_dofs_left::Int=1,
    n_dofs_right::Int=1,
)
    return create_bspline_space(
        starting_point,
        box_size,
        num_elements,
        Bernstein(degree),
        regularity;
        n_dofs_left=n_dofs_left,
        n_dofs_right=n_dofs_right,
    )
end

"""
    create_bspline_space(
        starting_point::NTuple{1, Float64},
        box_size::NTuple{1, Float64},
        num_elements::NTuple{1, Int},
        section_space::NTuple{1, F},
        regularity::NTuple{1, Int};
        n_dofs_left::NTuple{1, Int}=(1,),
        n_dofs_right::NTuple{1, Int}=(1,),
    ) where {F <: AbstractCanonicalSpace}

Create a 1D B-spline space based on the provided starting point, box size, number of
elements, section space, and regularity. It constructs a uniform distribution of
breakpoints and uses them to define a `Mesh.Patch1D` object. The function then creates a
regularity vector, with the first and last entries set to -1 to ensure an open knot vector,
and returns the corresponding `BSplineSpace`. Optional arguments include the number of
degrees of freedom on the left and right boundaries.

# Arguments
- `starting_point::NTuple{1, Float64}`: The coordinate of the starting point of the
    B-spline space.
- `box_size::NTuple{1, Float64}`: The size of the space along the single dimension.
- `num_elements::NTuple{1, Int}`: The number of elements along the dimension.
- `section_space::NTuple{1,F}`: The section space to use for the B-spline basis functions.
- `regularity::NTuple{1, Int}`: The regularity (smoothness) between consecutive B-spline
    basis functions, typically between 0 and `degree - 1`.
- `n_dofs_left::NTuple{1,Int}`: The number of degrees of freedom on the left boundary.
- `n_dofs_right::NTuple{1,Int}`: The number of degrees of freedom on the right boundary.

# Returns
- `::BSplineSpace`: The generated B-spline space.

# Examples
```julia
# Create a 1D B-spline space
start_1d = 0.0
size_1d = 10.0
elements_1d = 100
degree = 3
regularity = 2
section_space = Bernstein(degree)
bspline_space_1d = create_bspline_space((start_1d,), (size_1d,), (elements_1d,), (section_space,), (regularity,))
```
"""
function create_bspline_space(
    starting_point::NTuple{1, Float64},
    box_size::NTuple{1, Float64},
    num_elements::NTuple{1, Int},
    section_space::NTuple{1, F},
    regularity::NTuple{1, Int};
    n_dofs_left::NTuple{1, Int}=(1,),
    n_dofs_right::NTuple{1, Int}=(1,),
) where {F <: AbstractCanonicalSpace}
    return create_bspline_space(
        starting_point[1],
        box_size[1],
        num_elements[1],
        section_space[1],
        regularity[1];
        n_dofs_left=n_dofs_left[1],
        n_dofs_right=n_dofs_right[1],
    )
end

"""
    create_bspline_space(
        starting_point::NTuple{1, Float64},
        box_size::NTuple{1, Float64},
        num_elements::NTuple{1, Int},
        degree::NTuple{1, Int},
        regularity::NTuple{1, Int};
        n_dofs_left::NTuple{1, Int}=(1,),
        n_dofs_right::NTuple{1, Int}=(1,),
    )

Create a piecewise-polynomial 1D B-spline space based on the provided starting point,
box size, number of elements, polynomial degree, and regularity. It constructs a uniform
distribution of breakpoints and uses them to define a `Mesh.Patch1D` object. The function
then creates a regularity vector, with the first and last entries set to -1 to ensure an
open knot vector, and returns the corresponding `BSplineSpace`. Optional arguments include
the number of degrees of freedom on the left and right boundaries.

# Arguments
- `starting_point::Float64`: The coordinate of the starting point of the B-spline space.
- `box_size::Float64`: The size of the space along the single dimension.
- `num_elements::Int`: The number of elements along the dimension.
- `degree::Int`: The polynomial degree on each element.
- `regularity::Int`: The regularity (smoothness) between consecutive B-spline basis
    functions, typically between 0 and `degree - 1`.
- `n_dofs_left::Int`: The number of degrees of freedom on the left boundary.
- `n_dofs_right::Int`: The number of degrees of freedom on the right boundary.

# Returns
- `::BSplineSpace`: The generated B-spline space.
"""
function create_bspline_space(
    starting_point::NTuple{1, Float64},
    box_size::NTuple{1, Float64},
    num_elements::NTuple{1, Int},
    degree::NTuple{1, Int},
    regularity::NTuple{1, Int};
    n_dofs_left::NTuple{1, Int}=(1,),
    n_dofs_right::NTuple{1, Int}=(1,),
)
    return create_bspline_space(
        starting_point[1],
        box_size[1],
        num_elements[1],
        Bernstein(degree[1]),
        regularity[1];
        n_dofs_left=n_dofs_left[1],
        n_dofs_right=n_dofs_right[1],
    )
end

"""
    create_bspline_space(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        section_spaces::F,
        regularities::NTuple{manifold_dim, Int};
        n_dofs_left::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
        n_dofs_right::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
    ) where {manifold_dim, F <: NTuple{manifold_dim, AbstractCanonicalSpace}}

Create a tensor product B-spline space based on the specified parameters for each dimension.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: The starting point of the B-spline
    space in each dimension.
- `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box in each dimension.
- `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.
- `section_spaces::NTuple{manifold_dim, AbstractCanonicalSpace}`: The section spaces for
    each dimension.
- `regularities::NTuple{manifold_dim, Int}`: The regularities of the B-spline in each
    dimension.

# Returns
- `::TensorProductSpace`: The resulting tensor product B-spline space.
"""
function create_bspline_space(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    section_spaces::F,
    regularities::NTuple{manifold_dim, Int};
    n_dofs_left::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
    n_dofs_right::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
) where {manifold_dim, F <: NTuple{manifold_dim, AbstractCanonicalSpace}}
    return TensorProductSpace(
        create_dim_wise_bspline_spaces(
            starting_points,
            box_sizes,
            num_elements,
            section_spaces,
            regularities,
            n_dofs_left,
            n_dofs_right,
        ),
    )
end

"""
    create_dim_wise_bspline_spaces(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        section_spaces::F,
        regularities::NTuple{manifold_dim, Int},
        n_dofs_left::NTuple{manifold_dim, Int},
        n_dofs_right::NTuple{manifold_dim, Int},
    ) where {manifold_dim, F <: NTuple{manifold_dim, AbstractCanonicalSpace}}

Create `manifold_dim` univariate B-spline spaces based on the specified parameters for each
dimension.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: The starting point of the B-spline
    space in each dimension.
- `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box in each dimension.
- `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.
- `section_spaces::NTuple{manifold_dim, F}`: The section spaces for each dimension.
- `regularities::NTuple{manifold_dim, Int}`: The regularities of the B-spline in each
    dimension.

# Returns
- `::Tuple{BSplineSpace}`: The resulting univariate B-spline spaces.
"""
function create_dim_wise_bspline_spaces(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    section_spaces::F,
    regularities::NTuple{manifold_dim, Int},
    n_dofs_left::NTuple{manifold_dim, Int},
    n_dofs_right::NTuple{manifold_dim, Int},
) where {manifold_dim, F <: NTuple{manifold_dim, AbstractCanonicalSpace}}
    return ntuple(manifold_dim) do i
        create_bspline_space(
            starting_points[i],
            box_sizes[i],
            num_elements[i],
            section_spaces[i],
            regularities[i];
            n_dofs_left=n_dofs_left[i],
            n_dofs_right=n_dofs_right[i],
        )
    end
end

"""
    create_bspline_space(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        degrees::NTuple{manifold_dim, Int},
        regularities::NTuple{manifold_dim, Int};
        n_dofs_left::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
        n_dofs_right::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
    ) where {manifold_dim}

Create a tensor product B-spline space based on the specified parameters for each dimension.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: The starting point of the B-spline
    space in each dimension.
- `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box in each dimension.
- `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.
- `degrees::NTuple{manifold_dim, Int}`: The polynomial degree in each dimension.
- `regularities::NTuple{manifold_dim, Int}`: The regularities of the B-spline in each
    dimension.

# Returns
- `::TensorProductSpace`: The resulting tensor product B-spline space.
"""
function create_bspline_space(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    degrees::NTuple{manifold_dim, Int},
    regularities::NTuple{manifold_dim, Int};
    n_dofs_left::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
    n_dofs_right::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
) where {manifold_dim}
    return TensorProductSpace(
        create_dim_wise_bspline_spaces(
            starting_points,
            box_sizes,
            num_elements,
            degrees,
            regularities,
            n_dofs_left,
            n_dofs_right,
        ),
    )
end

"""
    create_dim_wise_bspline_spaces(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        degrees::NTuple{manifold_dim, Int},
        regularities::NTuple{manifold_dim, Int},
        n_dofs_left::NTuple{manifold_dim, Int},
        n_dofs_right::NTuple{manifold_dim, Int},
    ) where {manifold_dim}

Create `manifold_dim` univariate B-spline spaces based on the specified parameters for
each dimension.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: The starting point of the B-spline
    space in each dimension.
- `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box in each dimension.
- `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.
- `degrees::NTuple{manifold_dim, Int}`: The polynomial degree in each dimension.
- `regularities::NTuple{manifold_dim, Int}`: The regularities of the B-spline in each
    dimension.

# Returns
- `::Tuple{BSplineSpace}`: The resulting univariate B-spline spaces.
"""
function create_dim_wise_bspline_spaces(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    degrees::NTuple{manifold_dim, Int},
    regularities::NTuple{manifold_dim, Int},
    n_dofs_left::NTuple{manifold_dim, Int},
    n_dofs_right::NTuple{manifold_dim, Int},
) where {manifold_dim}
    return ntuple(manifold_dim) do i
        create_bspline_space(
            starting_points[i],
            box_sizes[i],
            num_elements[i],
            degrees[i],
            regularities[i];
            n_dofs_left=n_dofs_left[i],
            n_dofs_right=n_dofs_right[i],
        )
    end
end

################################################################################
# Polar spline helpers
################################################################################

"""
    create_scalar_polar_spline_space(
        num_elements::NTuple{2, Int},
        degrees::NTuple{2, Int},
        regularities::NTuple{2, Int};
        geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
        R::Float64 = 1.0,
        two_poles::Bool = false
    )

Create a scalar polar spline space and geometry based on the provided parameters. The
function constructs a tensor-product B-spline space in the angular and radial directions,
and then creates a polar spline space and geometry based on the tensor-product space. The
geometry is defined by the radius `R` and the number of elements in the angular and radial
directions. Optional arguments include a flag to refine the geometry, the geometry
coefficients, and whether the scalars are constrained to zero at poles.

# Arguments
- `num_elements::NTuple{2, Int}`: The number of elements in the angular and radial directions.
- `degrees::NTuple{2, Int}`: The polynomial degrees in the angular and radial directions.
- `regularities::NTuple{2, Int}`: The regularities of the B-spline in the angular and radial directions.
- `geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing`: The geometry coefficients for the polar spline space.
- `R::Float64=1.0`: The radius of the polar spline space.
- `two_poles::Bool=false`: Whether the polar spline space has two poles.
- `zero_at_poles::Bool=false`: Whether the scalars are constrained to zero at the poles.

# Returns
- `::PolarSplineSpace`: The resulting scalar polar spline space.
"""
function create_scalar_polar_spline_space(
    num_elements::NTuple{2, Int},
    degrees::NTuple{2, Int},
    regularities::NTuple{2, Int};
    geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
    R::Float64=1.0,
    two_poles::Bool=false,
    zero_at_poles::Bool=false,
    box_sizes::NTuple{2, Float64}=(1.0, 1.0),
)
    return create_scalar_polar_spline_space(
        num_elements,
        Bernstein.(degrees),
        regularities;
        geom_coeffs_tp=geom_coeffs_tp,
        R=R,
        two_poles=two_poles,
        zero_at_poles=zero_at_poles,
        box_sizes=box_sizes,
    )
end

"""
    create_scalar_polar_splines_space(
        num_elements::NTuple{2, Int},
        section_spaces::F,
        regularities::NTuple{2, Int};
        geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
        R::Float64=1.0,
        two_poles::Bool=false,
        zero_at_poles::Bool=false
    ) where {F <: NTuple{2, AbstractCanonicalSpace}}

Create a scalar polar spline space and geometry based on the provided parameters. The
function constructs a tensor-product B-spline space in the angular and radial directions,
and then creates a polar spline space and geometry based on the tensor-product space. The
geometry is defined by the radius `R` and the number of elements in the angular and radial
directions. Optional arguments include the geometry coefficients, whether there are two poles,
and whether the scalars are constrained to zero at the poles.

# Arguments
- `num_elements::NTuple{2, Int}`: The number of elements in the angular and radial directions.
- `section_spaces::F`: The section spaces for the angular and radial directions.
- `regularities::NTuple{2, Int}`: The regularities of the B-spline in the angular and radial directions.
- `geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing`: The geometry coefficients for the polar spline space.
- `R::Float64=1.0`: The radius of the polar spline space.
- `two_poles::Bool=false`: Whether the polar spline space has two poles.
- `zero_at_poles::Bool=false`: Whether the scalars are constrained to zero at the poles.

# Returns
- `::PolarSplineSpace`: The resulting scalar polar spline space.
"""
function create_scalar_polar_spline_space(
    num_elements::NTuple{2, Int},
    section_spaces::F,
    regularities::NTuple{2, Int};
    geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
    R::Float64=1.0,
    two_poles::Bool=false,
    zero_at_poles::Bool=false,
    box_sizes::NTuple{2, Float64}=(1.0, 1.0),
) where {F <: NTuple{2, AbstractCanonicalSpace}}
    # spaces used for creating degenerate geometry
    Bθ_g, Br_g = create_dim_wise_bspline_spaces(
        (0.0, 0.0), box_sizes, num_elements, section_spaces, regularities, (1, 1), (1, 1)
    )
    GBθ_g = GTBSplineSpace((Bθ_g,), [regularities[1]])

    if isnothing(geom_coeffs_tp)
        if two_poles
            error("Two-pole geometries require user-provided geometry coefficients.")
        end
        # number of control points for a degenerate tensor-product mapping
        n_θ = get_num_basis(GBθ_g)
        n_r = get_num_basis(Br_g)
        geom_coeffs_tp, _, _ = _build_standard_degenerate_control_points(n_θ, n_r, R)
    end

    if zero_at_poles
        # reduce degree and regularity by 1
        Bθ, Br = create_dim_wise_bspline_spaces(
            (0.0, 0.0),
            box_sizes,
            num_elements,
            get_derivative_space.(section_spaces),
            regularities .- 1,
            (1, 1),
            (1, 1),
        )
        GBθ = GTBSplineSpace((Bθ,), [regularities[1] - 1])

        return PolarSplineSpace(
            (TensorProductSpace((GBθ, Br)),),
            geom_coeffs_tp,
            TensorProductSpace((GBθ_g, Br_g)),
            two_poles,
            zero_at_poles,
        )
    else
        return PolarSplineSpace(
            (TensorProductSpace((GBθ_g, Br_g)),),
            geom_coeffs_tp,
            TensorProductSpace((GBθ_g, Br_g)),
            two_poles,
            zero_at_poles,
        )
    end
end

"""
    create_vector_polar_spline_space(
        num_elements::NTuple{2, Int},
        degrees::NTuple{2, Int},
        regularities::NTuple{2, Int};
        geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
        R::Float64=1.0,
        two_poles::Bool=false
    )

Create a vector polar spline space and geometry based on the provided parameters. The
function constructs two tensor-product B-spline spaces, and then creates a 2-component polar
spline space and geometry based on them. The geometry is defined by the radius `R`
and the number of elements in the angular and radial directions. Optional arguments include
the geometry coefficients, and whether there are two poles.

# Arguments
- `num_elements::NTuple{2, Int}`: The number of elements in each direction.
- `degrees::NTuple{2, Int}`: The polynomial degrees in each direction.
- `regularities::NTuple{2, Int}`: The regularities in each direction.
- `geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}`: The geometry coefficients.
- `R::Float64`: The radius of the domain.

# Returns
- `::PolarSplineSpace`: The resulting vector polar spline space.
"""
function create_vector_polar_spline_space(
    num_elements::NTuple{2, Int},
    degrees::NTuple{2, Int},
    regularities::NTuple{2, Int};
    geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
    R::Float64=1.0,
    two_poles::Bool=false,
    box_sizes::NTuple{2, Float64}=(1.0, 1.0),
)
    return create_vector_polar_spline_space(
        num_elements,
        Bernstein.(degrees),
        regularities;
        geom_coeffs_tp=geom_coeffs_tp,
        R=R,
        two_poles=two_poles,
        box_sizes=box_sizes,
    )
end

"""
    create_vector_polar_spline_space(
        num_elements::NTuple{2, Int},
        section_spaces::F,
        regularities::NTuple{2, Int};
        geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
        R::Float64=1.0,
        two_poles::Bool=false,
    ) where {F <: NTuple{2, AbstractCanonicalSpace}}

Create a vector polar spline space and geometry based on the provided parameters. The
function constructs two tensor-product B-spline spaces, and then creates a 2-component polar
spline space and geometry based on them. The geometry is defined by the radius `R`
and the number of elements in the angular and radial directions. Optional arguments include
the geometry coefficients, and whether there are two poles.

# Arguments
- `num_elements::NTuple{2, Int}`: The number of elements in each direction.
- `degrees::NTuple{2, Int}`: The polynomial degrees in each direction.
- `regularities::NTuple{2, Int}`: The regularities in each direction.
- `geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}`: The geometry coefficients.
- `R::Float64`: The radius of the domain.

# Returns
- `::PolarSplineSpace`: The resulting vector polar spline space.
"""
function create_vector_polar_spline_space(
    num_elements::NTuple{2, Int},
    section_spaces::F,
    regularities::NTuple{2, Int};
    geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
    R::Float64=1.0,
    two_poles::Bool=false,
    box_sizes::NTuple{2, Float64}=(1.0, 1.0),
) where {F <: NTuple{2, AbstractCanonicalSpace}}
    # spaces used for creating degenerate geometry
    Bθ_g, Br_g = create_dim_wise_bspline_spaces(
        (0.0, 0.0), box_sizes, num_elements, section_spaces, regularities, (1, 1), (1, 1)
    )
    GBθ_g = GTBSplineSpace((Bθ_g,), [regularities[1]])

    if isnothing(geom_coeffs_tp)
        if two_poles
            error("Two-pole geometries require user-provided geometry coefficients.")
        end
        # number of control points for a degenerate tensor-product mapping
        n_θ = get_num_basis(GBθ_g)
        n_r = get_num_basis(Br_g)
        geom_coeffs_tp, _, _ = _build_standard_degenerate_control_points(n_θ, n_r, R)
    end

    return PolarSplineSpace(
        (
            TensorProductSpace((get_derivative_space(GBθ_g), Br_g)),
            TensorProductSpace((GBθ_g, get_derivative_space(Br_g))),
        ),
        geom_coeffs_tp,
        TensorProductSpace((GBθ_g, Br_g)),
        two_poles,
        false,
    )
end

"""
    create_polar_geometry_data(
        num_elements::NTuple{2, Int},
        degrees::NTuple{2, Int},
        regularities::NTuple{2, Int};
        geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
        R::Float64=1.0,
        two_poles::Bool=false
    )

Create all data required for creating a polar spline geometry.

# Arguments
- `num_elements::NTuple{2, Int}`: The number of elements in each direction.
- `degrees::NTuple{2, Int}`: The polynomial degrees in each direction.
- `regularities::NTuple{2, Int}`: The regularities in each direction.
- `geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}`: The geometry coefficients.
- `R::Float64`: The radius of the domain.

# Returns
- `P_geom`: The polar spline geometry space.
- `geom_coeffs_polar`: The polar geometry coefficients.
"""
function create_polar_geometry_data(
    num_elements::NTuple{2, Int},
    degrees::NTuple{2, Int},
    regularities::NTuple{2, Int};
    geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
    R::Float64=1.0,
    two_poles::Bool=false,
    box_sizes::NTuple{2, Float64}=(1.0, 1.0),
)
    return create_polar_geometry_data(
        num_elements,
        Bernstein.(degrees),
        regularities;
        geom_coeffs_tp=geom_coeffs_tp,
        R=R,
        two_poles=two_poles,
        box_sizes=box_sizes,
    )
end

function create_polar_geometry_data(
    num_elements::NTuple{2, Int},
    section_spaces::F,
    regularities::NTuple{2, Int};
    geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
    R::Float64=1.0,
    two_poles::Bool=false,
    box_sizes::NTuple{2, Float64}=(1.0, 1.0),
) where {F <: NTuple{2, AbstractCanonicalSpace}}
    Bθ, Br = create_dim_wise_bspline_spaces(
        (0.0, 0.0), box_sizes, num_elements, section_spaces, regularities, (1, 1), (1, 1)
    )
    # impose periodicity
    GBθ = GTBSplineSpace((Bθ,), [regularities[1]])

    # number of control points for a degenerate tensor-product mapping
    n_θ = get_num_basis(GBθ)
    n_r = get_num_basis(Br)

    if isnothing(geom_coeffs_tp)
        geom_coeffs_tp, _, _ = _build_standard_degenerate_control_points(n_θ, n_r, R)
    end

    # Polar spline space and global extraction matrix for the geometry
    P_geom = PolarSplineSpace(
        (TensorProductSpace((GBθ, Br)),),
        geom_coeffs_tp,
        TensorProductSpace((GBθ, Br)),
        two_poles,
        false,
    )
    E_geom = assemble_global_extraction_matrix(P_geom)

    # control points for the polar spline space
    geom_coeffs_polar = (E_geom' * E_geom) \ (E_geom' * reshape(geom_coeffs_tp, :, 2))

    return P_geom, geom_coeffs_polar
end

"""
    refine_polar_geometry_data(P_geom, geom_coeffs_polar; two_poles=false)

Refine the polar geometry data.

# Arguments
- `P_geom`: The polar spline geometry space.
- `geom_coeffs_polar`: The polar geometry coefficients.
- `two_poles`: Whether the polar geometry contains two poles.

# Returns
- `P_geom_ref`: The refined polar spline geometry space.
- `geom_coeffs_polar_tp_ref`: The refined polar geometry coefficients.
"""
function refine_polar_geometry_data(P_geom, geom_coeffs_polar; two_poles=false)
    # refine the space and build two-scale operator
    TS, P_geom_ref = build_two_scale_operator(P_geom, 2)
    subdiv_mat = get_global_subdiv_matrix(TS)
    # refine input polar geometry coefficients
    geom_coeffs_polar_ref = subdiv_mat * geom_coeffs_polar

    return P_geom_ref,
    geom_coeffs_polar_ref,
    get_num_elements.(get_constituent_spaces(P_geom_ref))
end

"""
    _build_standard_degenerate_control_points(n_p::Int, n_r::Int, R::Float64)

Generate degenerate control points for a given number of poloidal and radial divisions.

# Arguments
- n_p::Int: Number of poloidal divisions.
- n_r::Int: Number of radial divisions.
- R::Float64: Radius of the polar domain.

# Returns
- degenerate_control_points::Array{Float64,3}: The degenerate control points.
- radii::Vector{Float64}: The radii values.
- theta::Vector{Float64}: The theta values.
"""
function _build_standard_degenerate_control_points(n_p::Int, n_r::Int, R::Float64)
    radii = LinRange(0.0, R, n_r)
    theta = 2 * pi .+ (1 .- 2 .* (1:n_p)) .* pi / n_p
    degenerate_control_points = zeros(Float64, n_p, n_r, 2)
    # the control points that are identical to the tensor-product control points
    for idx in Iterators.product(1:n_p, 1:n_r)
        degenerate_control_points[idx[1], idx[2], 1] = radii[idx[2]] * cos(theta[idx[1]])
        degenerate_control_points[idx[1], idx[2], 2] = radii[idx[2]] * sin(theta[idx[1]])
    end

    return degenerate_control_points, radii, theta
end

################################################################################
# Masking for hierarchical spaces
################################################################################

function create_evaluation_mask_from_hierarchical_mesh(
    hierarchical_space::FunctionSpaces.HierarchicalFiniteElementSpace{manifold_dim, S, T};
    exclude_elements::Vector{Int}=Int[],
) where {
    manifold_dim,
    S <: FunctionSpaces.AbstractFESpace{manifold_dim, 1},
    T <: FunctionSpaces.AbstractTwoScaleOperator,
}
    # initialize trivial evaluation mask for first level space
    base_space = FunctionSpaces.get_space(hierarchical_space, 1)
    # number of elements in base mesh
    num_elements_base = FunctionSpaces.get_num_elements(base_space)
    # get element vertices for the base space
    base_element_vertices = [
        FunctionSpaces.get_element_vertices(base_space, i) for i in 1:num_elements_base
    ]

    # number of elements in evaluation mask (i.e., the active hierarchical mesh)
    num_elements = FunctionSpaces.get_num_elements(hierarchical_space)
    # get eval to base element index map
    element_idx_map = zeros(Int64, num_elements)
    # get element vertices for the eval mask
    eval_element_vertices = Vector{NTuple{manifold_dim, Vector{Float64}}}(
        undef, num_elements
    )
    for i in 1:num_elements
        element_level, element_level_id = FunctionSpaces.convert_to_element_level_and_level_id(
            hierarchical_space, i
        )
        eval_element_vertices[i] = FunctionSpaces.get_element_vertices(
            FunctionSpaces.get_space(hierarchical_space, element_level), element_level_id
        )
        element_idx_map[i] = FunctionSpaces.get_element_ancestor(
            hierarchical_space.two_scale_operators,
            element_level_id,
            element_level,
            element_level - 1,
        )
    end

    # allocate memory for translations and scalings
    translations = Vector{NTuple{manifold_dim, Float64}}(undef, num_elements)
    scalings = Vector{NTuple{manifold_dim, Float64}}(undef, num_elements)

    # create length scales (i.e., ratios of child and ancestor)
    for element_idx in 1:num_elements
        element_idx_base = element_idx_map[element_idx]
        base_el_verts = base_element_vertices[element_idx_base]
        eval_el_verts = eval_element_vertices[element_idx]

        length_scales =
            [eval_el_verts[k][2] - eval_el_verts[k][1] for k in 1:manifold_dim] ./ [base_el_verts[k][2] - base_el_verts[k][1] for k in 1:manifold_dim]
        scalings[element_idx] = tuple(length_scales...)

        translations[element_idx] = tuple(
            [
                (eval_el_verts[k][1] - base_el_verts[k][1]) /
                (base_el_verts[k][2] - base_el_verts[k][1]) for k in 1:manifold_dim
            ]...,
        )
    end

    # create corresponding evaluation mask
    E = EvaluationMask.AffineEvaluationMask(
        num_elements, num_elements_base, element_idx_map, translations, scalings
    )
    # return trimmed mask
    return EvaluationMask.trim_evaluation_mask(E, exclude_elements)
end

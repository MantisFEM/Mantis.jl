function subdivide_geometry(parent_geo::Geometry.MappedGeometry, num_subdivisons)
    parent_base_geometry = Geometry.get_base_geometry(parent_geo)
    child_base_geoemtry = subdivide_geometry(parent_base_geometry, num_subdivisons)
    mapping = Geometry.get_mapping(parent_geo)
    child_geometry = Geometry.MappedGeometry(child_base_geoemtry, mapping)

    return child_geometry
end

function subdivide_geometry(parent_geo::Geometry.CartesianGeometry{1}, num_subdivisons)
    parent_breakpoints = Geometry.get_breakpoints(parent_geo)[1]
    child_breakpoints = subdivide_breakpoints(parent_breakpoints, num_subdivisons)
    child_geometry = Geometry.CartesianGeometry(child_breakpoints)

    return child_geometry
end

"""
    subdivide_breakpoints(parent_breakpoints::Vector{Float64}, num_subdivisions::Int)

Subdivides `parent_breakpoints` by uniformly subdiving each element `num_subdivisions`
times.

# Arguments
- `parent_breakpoints::AbstractVector`: Parent set of breakpoints.
- `num_subdivisions`: Number of times each element is subdivided.
# Returns
- `child_breakpoints::Vector{Float64}`: Child set of breakpoints.
"""
function subdivide_breakpoints(parent_breakpoints::AbstractVector, num_subdivisions)
    num_parent_breakpoints = length(parent_breakpoints)
    num_child_breakpoints =
        num_parent_breakpoints + (num_parent_breakpoints - 1) * (num_subdivisions - 1)
    child_breakpoints = Vector{Float64}(undef, num_child_breakpoints)
    child_breakpoints[end] = parent_breakpoints[end]
    step_size = 1 / num_subdivisions
    for i in 1:(num_parent_breakpoints - 1), j in 0:(num_subdivisions - 1)
        index = (i - 1) * num_subdivisions + j + 1
        child_breakpoints[index] =
            parent_breakpoints[i] +
            j * step_size * (parent_breakpoints[i + 1] - parent_breakpoints[i])
    end

    return child_breakpoints
end

function subdivide_breakpoints(parent_breakpoints::LinRange, num_subdivisions)
    start_breakpoint = first(parent_breakpoints)
    end_breakpoint = last(parent_breakpoints)
    parent_num_elements = length(parent_breakpoints) - 1
    child_breakpoints = LinRange(
        start_breakpoint,
        end_breakpoint,
        (length(parent_breakpoints) - 1) * num_subdivisions + 1,
    )

    return child_breakpoints
end

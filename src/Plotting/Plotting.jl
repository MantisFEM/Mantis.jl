struct VTKData
    n_points::Int
    points::Matrix{Float64}
    n_cells::Int
    cells::Matrix{Int}
    cell_type::Int
    point_data::Vector{Float64}
end
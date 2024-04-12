# This is the fuction I used in my code to extract vtk data 
#=
function extract_vtk_data(mixedSpace, el_errors, el_curl_errors, u_coeffs, s_coeffs)
    n_quad = mixedSpace.zeroFormSpace.refData.n_quad
    n_points = n_quad^2
    n_cells = Int((n_quad-1)^2)
    
    VTK_n_points = n_points * mixedSpace.mesh.n_els
    VTK_n_cells = n_cells * mixedSpace.mesh.n_els

    VTK_points = zeros((VTK_n_points, 3))
    VTK_cells = zeros(Int, (VTK_n_cells, 4))

    VTK_point_data = zeros(VTK_n_points)
    VTK_vector_data = zeros((VTK_n_points,3))
    VTK_l2_cell_data = zeros(VTK_n_cells)
    VTK_curl_cell_data = zeros(VTK_n_cells)

    dof_offset = mixedSpace.oneFormSpace.scalarSpaces[1].dimension

    ref_cell = hcat([AuxFuncs.extract_cell_points(i, n_quad) for i = 1:n_cells]...)'

    N_u, dN_u, N_s, dN_s, I_u, I_s = AuxFuncs.init_data_matrices(mixedSpace)
    u_val = zeros((3, n_quad, n_quad))
    s_val = [zeros((3, n_quad, n_quad)), zeros((3, n_quad, n_quad))]

    points = zeros(n_points,2)

    #solution_u((x,y)) = 16*x*(1-x)*y*(1-y) #sin(x)*sin(y) #

    count = 0
    for el in 1:mixedSpace.mesh.n_els
        AuxFuncs.fill_value_zeros!(u_val, s_val)
        AuxFuncs.fill_data_zeros!(N_u, dN_u, N_s, dN_s)

        x, dxi, J = AuxFuncs.evaluate_geometry(mixedSpace.geo, el)

        I_u, N_u, dN_u = AuxFuncs.extract_el_zero_data!(mixedSpace, el, dxi, I_u, N_u, dN_u)
        I_s, N_s, dN_s = AuxFuncs.extract_el_one_data!(mixedSpace, el, dxi, I_s, N_s, dN_s)

        u_val = AuxFuncs.el_z_values!(u_val, I_u, N_u, dN_u, u_coeffs)
        s_val = (AuxFuncs.el_z_values!(s_val[1], I_s[1], N_s[1], dN_s[1], s_coeffs[1:dof_offset]), AuxFuncs.el_z_values!(s_val[2], I_s[2], N_s[2], dN_s[2], s_coeffs[dof_offset+1:end]))

        p_count = 1
        for j in 1:n_quad,i in 1:n_quad
            points[p_count,:] .= x[i,j]
            p_count += 1
        end

        VTK_points[n_points*count+1:n_points*(count+1), 1:2] .= points
        VTK_points[n_points*count+1:n_points*(count+1), 3] .= reshape(u_val[1,:,:], n_points)
        VTK_cells[n_cells*count+1:n_cells*(count+1),:] .= ref_cell .+ n_points*count
        VTK_point_data[n_points*count+1:n_points*(count+1)] .= reshape(u_val[1,:,:], n_points)
        VTK_vector_data[n_points*count+1:n_points*(count+1), 1] .= reshape(s_val[1][1,:,:], n_points)
        VTK_vector_data[n_points*count+1:n_points*(count+1), 2] .= reshape(s_val[2][1,:,:], n_points)
        VTK_l2_cell_data[n_cells*count+1:n_cells*(count+1)] .= el_errors[el]
        VTK_curl_cell_data[n_cells*count+1:n_cells*(count+1)] .= el_curl_errors[el]
        count += 1
    end

    return Structures.VTKData(VTK_n_points, VTK_points, VTK_n_cells, VTK_cells, VTK_point_data, VTK_l2_cell_data, VTK_curl_cell_data, VTK_vector_data)
end
=#

struct VTKData
    n_points::Int
    points::Matrix{Float64}
    n_cells::Int
    cells::Matrix{Int}
    cell_type::Int
    point_data::Vector{Float64}
end
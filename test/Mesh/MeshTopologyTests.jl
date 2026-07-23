import Mantis

using Test

# -----------------------------------------------------------------------------
# Test conversion from position to id 
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Test 1D
# -----------------------------------------------------------------------------
# Test vertices
for vertex_id in 1:2
    @test vertex_id == Mantis.Mesh.position2id(Mantis.Mesh.id2position(1, 0, vertex_id))
end

# Test edges
for edge_id in 1:1
    @test edge_id == Mantis.Mesh.position2id(Mantis.Mesh.id2position(1, 1, edge_id))
end

# -----------------------------------------------------------------------------
# Test 2D
# -----------------------------------------------------------------------------
# Test vertices
for vertex_id in 1:4
    @test vertex_id == Mantis.Mesh.position2id(Mantis.Mesh.id2position(2, 0, vertex_id))
end

# Test edges
for edge_id in 1:4
    @test edge_id == Mantis.Mesh.position2id(Mantis.Mesh.id2position(2, 1, edge_id))
end

# Test surfaces
for surface_id in 1:1
    @test surface_id == Mantis.Mesh.position2id(Mantis.Mesh.id2position(2, 2, surface_id))
end

# -----------------------------------------------------------------------------
# Test 3D
# -----------------------------------------------------------------------------
# Test vertices
for vertex_id in 1:8
    @test vertex_id == Mantis.Mesh.position2id(Mantis.Mesh.id2position(3, 0, vertex_id))
end

# Test edges
for edge_id in 1:12
    @test edge_id == Mantis.Mesh.position2id(Mantis.Mesh.id2position(3, 1, edge_id))
end

# Test surfaces
for surface_id in 1:6
    @test surface_id == Mantis.Mesh.position2id(Mantis.Mesh.id2position(3, 2, surface_id))
end

# Test volumes
for volume_id in 1:1
    @test volume_id == Mantis.Mesh.position2id(Mantis.Mesh.id2position(3, 3, volume_id))
end
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Test 3D mesh topology 
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Test mesh 1: 2 patches, configuration 1
# -----------------------------------------------------------------------------
#   1st patch: 1, 2, 3, 4, 5, 6, 7, 8      (vertex indices)
#   2nd patch: 7, 6, 2, 3, 12, 11, 9, 10   (vertex indices)

# Define the mesh
mesh_connectivity_3d_ex_1 = [[1, 2, 3, 4, 5, 6, 7, 8], [7, 6, 2, 3, 12, 11, 9, 10]]

# Define reference results
# Incidence relations
incidence_relation_ex_1_ref = Matrix{Vector{Vector{Int}}}(undef, 4, 4)
incidence_relation_ex_1_ref[1, 1] = Vector{Int64}[]
incidence_relation_ex_1_ref[1, 2] = [
    [1, 2, 3],
    [1, 4, 5, 6],
    [4, 7, 8, 9],
    [2, 7, 10],
    [3, 11, 12],
    [5, 11, 13, 14],
    [8, 13, 15, 16],
    [10, 12, 15],
    [6, 17, 18],
    [9, 17, 19],
    [14, 18, 20],
    [16, 19, 20],
]
incidence_relation_ex_1_ref[1, 3] = [
    [1, 2, 3],
    [1, 2, 4, 5, 6],
    [1, 4, 5, 7, 8],
    [1, 3, 7],
    [2, 3, 9],
    [2, 4, 6, 9, 10],
    [4, 7, 8, 9, 10],
    [3, 7, 9],
    [5, 6, 11],
    [5, 8, 11],
    [6, 10, 11],
    [8, 10, 11],
]
incidence_relation_ex_1_ref[1, 4] = [
    [1], [1, 2], [1, 2], [1], [1], [1, 2], [1, 2], [1], [2], [2], [2], [2]
]
incidence_relation_ex_1_ref[2, 1] = [
    [2, 1],
    [4, 1],
    [5, 1],
    [2, 3],
    [6, 2],
    [9, 2],
    [3, 4],
    [7, 3],
    [10, 3],
    [8, 4],
    [6, 5],
    [8, 5],
    [7, 6],
    [11, 6],
    [7, 8],
    [12, 7],
    [9, 10],
    [9, 11],
    [10, 12],
    [11, 12],
]
incidence_relation_ex_1_ref[2, 2] = Vector{Int64}[]
incidence_relation_ex_1_ref[2, 3] = [
    [1, 2],
    [1, 3],
    [2, 3],
    [1, 4, 5],
    [2, 4, 6],
    [5, 6],
    [1, 7],
    [4, 7, 8],
    [5, 8],
    [3, 7],
    [2, 9],
    [3, 9],
    [4, 9, 10],
    [6, 10],
    [7, 9],
    [8, 10],
    [5, 11],
    [6, 11],
    [8, 11],
    [10, 11],
]
incidence_relation_ex_1_ref[2, 4] = [
    [1],
    [1],
    [1],
    [1, 2],
    [1, 2],
    [2],
    [1],
    [1, 2],
    [2],
    [1],
    [1],
    [1],
    [1, 2],
    [2],
    [1],
    [2],
    [2],
    [2],
    [2],
    [2],
]
incidence_relation_ex_1_ref[3, 1] = [
    [3, 2, 1, 4],
    [6, 5, 1, 2],
    [8, 4, 1, 5],
    [7, 6, 2, 3],
    [9, 2, 3, 10],
    [9, 11, 6, 2],
    [7, 3, 4, 8],
    [10, 3, 7, 12],
    [7, 8, 5, 6],
    [11, 12, 7, 6],
    [9, 10, 12, 11],
]
incidence_relation_ex_1_ref[3, 2] = [
    [7, 1, -4, 2],
    [5, 3, 11, 1],
    [12, 2, 10, 3],
    [8, 5, 13, -4],
    [17, 4, 6, 9],
    [6, 14, 18, -5],
    [15, 7, 8, 10],
    [19, -8, 9, 16],
    [13, 12, 15, 11],
    [14, 16, 20, -13],
    [18, 19, 17, 20],
]
incidence_relation_ex_1_ref[3, 3] = Vector{Int64}[]
incidence_relation_ex_1_ref[3, 4] = [
    [1], [1], [1], [1, 2], [2], [2], [1], [2], [1], [2], [2]
]
incidence_relation_ex_1_ref[4, 1] = [[1, 2, 3, 4, 5, 6, 7, 8], [7, 6, 2, 3, 12, 11, 9, 10]]
incidence_relation_ex_1_ref[4, 2] = [
    [-15, -11, -1, -7, -13, -12, -2, 4, -8, -10, -3, -5],
    [-17, -20, 13, -4, -18, -19, 8, 5, -6, -9, -16, -14],
]
incidence_relation_ex_1_ref[4, 3] = [[3, 4, 2, 7, 1, 9], [8, 6, 10, 5, -4, 11]]
incidence_relation_ex_1_ref[4, 4] = Vector{Int64}[]

# Face neighbours 
n_total_patches = size(mesh_connectivity_3d_ex_1, 1)
n_local_faces = 6  # hexahedra
face_neighbours_3d_ex_1_ref = Array{Matrix{Int}}(undef, n_total_patches, n_local_faces)

face_neighbours_3d_ex_1_ref[1, 1] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_1_ref[1, 2] = [2; 5; 2; 1;;]
face_neighbours_3d_ex_1_ref[1, 3] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_1_ref[1, 4] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_1_ref[1, 5] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_1_ref[1, 6] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_1_ref[2, 1] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_1_ref[2, 2] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_1_ref[2, 3] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_1_ref[2, 4] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_1_ref[2, 5] = [1; 2; 2; 1;;]
face_neighbours_3d_ex_1_ref[2, 6] = Matrix{Int64}(undef, 4, 0)

# Edge neighbours 
n_total_patches = size(mesh_connectivity_3d_ex_1, 1)
n_local_edges = 12  # hexahedra
edge_neighbours_3d_ex_1_ref = Array{Matrix{Int}}(undef, n_total_patches, n_local_edges)

edge_neighbours_3d_ex_1_ref[1, 1] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[1, 2] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[1, 3] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[1, 4] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[1, 5] = [2; 3; 0; -1;;]
edge_neighbours_3d_ex_1_ref[1, 6] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[1, 7] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[1, 8] = [2; 4; 0; -1;;]
edge_neighbours_3d_ex_1_ref[1, 9] = [2; 7; 0; -1;;]
edge_neighbours_3d_ex_1_ref[1, 10] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[1, 11] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[1, 12] = [2; 8; 0; -1;;]
edge_neighbours_3d_ex_1_ref[2, 1] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[2, 2] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[2, 3] = [1; 5; 0; -1;;]
edge_neighbours_3d_ex_1_ref[2, 4] = [1; 8; 0; -1;;]
edge_neighbours_3d_ex_1_ref[2, 5] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[2, 6] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[2, 7] = [1; 9; 0; -1;;]
edge_neighbours_3d_ex_1_ref[2, 8] = [1; 12; 0; -1;;]
edge_neighbours_3d_ex_1_ref[2, 9] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[2, 10] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[2, 11] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_1_ref[2, 12] = Matrix{Int64}(undef, 4, 0)

# Vertex neighbours 
n_total_patches = size(mesh_connectivity_3d_ex_1, 1)
n_local_vertices = 8  # hexahedra
vertex_neighbours_3d_ex_1_ref = Array{Matrix{Int}}(undef, n_total_patches, n_local_vertices)

vertex_neighbours_3d_ex_1_ref[1, 1] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_1_ref[1, 2] = [2; 3; 0; 0;;]
vertex_neighbours_3d_ex_1_ref[1, 3] = [2; 4; 0; 0;;]
vertex_neighbours_3d_ex_1_ref[1, 4] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_1_ref[1, 5] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_1_ref[1, 6] = [2; 2; 0; 0;;]
vertex_neighbours_3d_ex_1_ref[1, 7] = [2; 1; 0; 0;;]
vertex_neighbours_3d_ex_1_ref[1, 8] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_1_ref[2, 1] = [1; 7; 0; 0;;]
vertex_neighbours_3d_ex_1_ref[2, 2] = [1; 6; 0; 0;;]
vertex_neighbours_3d_ex_1_ref[2, 3] = [1; 2; 0; 0;;]
vertex_neighbours_3d_ex_1_ref[2, 4] = [1; 3; 0; 0;;]
vertex_neighbours_3d_ex_1_ref[2, 5] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_1_ref[2, 6] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_1_ref[2, 7] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_1_ref[2, 8] = Matrix{Int64}(undef, 4, 0)

# Test MeshTopology
# Compute the mesh topology
mesh_topology_3d_ex_1 = Mantis.Mesh.MeshTopology(mesh_connectivity_3d_ex_1)
# Check if incidence relations are correct 
for i in 1:4
    for j in 1:4
        @test incidence_relation_ex_1_ref[i, j] == mesh_topology_3d_ex_1[i, j]
    end
end

# Test face neighbours
# Compute face neighbours 
face_neighbours_3d_ex_1 = Mantis.Mesh.compute_face_neighbours(mesh_topology_3d_ex_1)
# Check if face neighbours are correct 
for i in 1:n_total_patches
    for j in 1:n_local_faces
        @test face_neighbours_3d_ex_1_ref[i, j] == face_neighbours_3d_ex_1[i, j]
    end
end

# Test edge neighbours
# Compute edge neighbours 
edge_neighbours_3d_ex_1 = Mantis.Mesh.compute_edge_neighbours(mesh_topology_3d_ex_1)
# Check if edge neighbours are correct 
for i in 1:n_total_patches
    for j in 1:n_local_edges
        @test edge_neighbours_3d_ex_1_ref[i, j] == edge_neighbours_3d_ex_1[i, j]
    end
end

# Test vertex neighbours
# Compute vertex neighbours 
vertex_neighbours_3d_ex_1 = Mantis.Mesh.compute_vertex_neighbours(mesh_topology_3d_ex_1)
# Check if edge neighbours are correct 
for i in 1:n_total_patches
    for j in 1:n_local_vertices
        @test vertex_neighbours_3d_ex_1_ref[i, j] == vertex_neighbours_3d_ex_1[i, j]
    end
end

# -----------------------------------------------------------------------------
# Test mesh 2: 2 patches, configuration 2
# -----------------------------------------------------------------------------
#   1st patch: 1, 2, 3, 4, 5, 6, 7, 8      (vertex indices)
#   2nd patch: 10, 3, 2, 9, 12, 7, 6, 11   (vertex indices)

# Define the mesh
mesh_connectivity_3d_ex_2 = [[1, 2, 3, 4, 5, 6, 7, 8], [10, 3, 2, 9, 12, 7, 6, 11]]

# Define reference results
# Incidence relations
incidence_relation_ex_2_ref = Matrix{Vector{Vector{Int}}}(undef, 4, 4)
incidence_relation_ex_2_ref[1, 1] = Vector{Int64}[]
incidence_relation_ex_2_ref[1, 2] = [
    [1, 2, 3],
    [1, 4, 5, 6],
    [4, 7, 8, 9],
    [2, 7, 10],
    [3, 11, 12],
    [5, 11, 13, 14],
    [8, 13, 15, 16],
    [10, 12, 15],
    [6, 17, 18],
    [9, 17, 19],
    [14, 18, 20],
    [16, 19, 20],
]
incidence_relation_ex_2_ref[1, 3] = [
    [1, 2, 3],
    [1, 2, 4, 5, 6],
    [1, 4, 5, 7, 8],
    [1, 3, 7],
    [2, 3, 9],
    [2, 4, 6, 9, 10],
    [4, 7, 8, 9, 10],
    [3, 7, 9],
    [5, 6, 11],
    [5, 8, 11],
    [6, 10, 11],
    [8, 10, 11],
]
incidence_relation_ex_2_ref[1, 4] = [
    [1], [1, 2], [1, 2], [1], [1], [1, 2], [1, 2], [1], [2], [2], [2], [2]
]
incidence_relation_ex_2_ref[2, 1] = [
    [2, 1],
    [4, 1],
    [5, 1],
    [3, 2],
    [6, 2],
    [2, 9],
    [3, 4],
    [7, 3],
    [3, 10],
    [8, 4],
    [6, 5],
    [8, 5],
    [7, 6],
    [6, 11],
    [7, 8],
    [7, 12],
    [9, 10],
    [11, 9],
    [12, 10],
    [11, 12],
]
incidence_relation_ex_2_ref[2, 2] = Vector{Int64}[]
incidence_relation_ex_2_ref[2, 3] = [
    [1, 2],
    [1, 3],
    [2, 3],
    [1, 4, 5],
    [2, 4, 6],
    [5, 6],
    [1, 7],
    [4, 7, 8],
    [5, 8],
    [3, 7],
    [2, 9],
    [3, 9],
    [4, 9, 10],
    [6, 10],
    [7, 9],
    [8, 10],
    [5, 11],
    [6, 11],
    [8, 11],
    [10, 11],
]
incidence_relation_ex_2_ref[2, 4] = [
    [1],
    [1],
    [1],
    [1, 2],
    [1, 2],
    [2],
    [1],
    [1, 2],
    [2],
    [1],
    [1],
    [1],
    [1, 2],
    [2],
    [1],
    [2],
    [2],
    [2],
    [2],
    [2],
]
incidence_relation_ex_2_ref[3, 1] = [
    [3, 2, 1, 4],
    [6, 5, 1, 2],
    [8, 4, 1, 5],
    [7, 6, 2, 3],
    [2, 3, 10, 9],
    [6, 2, 9, 11],
    [7, 3, 4, 8],
    [7, 12, 10, 3],
    [7, 8, 5, 6],
    [6, 11, 12, 7],
    [11, 9, 10, 12],
]
incidence_relation_ex_2_ref[3, 2] = [
    [7, 1, 4, 2],
    [5, 3, 11, 1],
    [12, 2, 10, 3],
    [8, 5, 13, 4],
    [6, 9, -4, 17],
    [14, 6, 5, 18],
    [15, 7, 8, 10],
    [8, 19, 16, 9],
    [13, 12, 15, 11],
    [-13, 20, 14, 16],
    [20, 17, 18, 19],
]
incidence_relation_ex_2_ref[3, 3] = Vector{Int64}[]
incidence_relation_ex_2_ref[3, 4] = [
    [1], [1], [1], [1, 2], [2], [2], [1], [2], [1], [2], [2]
]
incidence_relation_ex_2_ref[4, 1] = [[1, 2, 3, 4, 5, 6, 7, 8], [10, 3, 2, 9, 12, 7, 6, 11]]
incidence_relation_ex_2_ref[4, 2] = [
    [-15, -11, -1, -7, -13, -12, -2, -4, -8, -10, -3, -5],
    [-14, -16, -9, -6, 13, -20, -17, 4, -5, -18, -19, -8],
]
incidence_relation_ex_2_ref[4, 3] = [[3, 4, 2, 7, 1, 9], [11, -4, 8, 6, 5, 10]]
incidence_relation_ex_2_ref[4, 4] = Vector{Int64}[]

# Face neighbours 
n_total_patches = size(mesh_connectivity_3d_ex_2, 1)
n_local_faces = 6  # hexahedra
face_neighbours_3d_ex_2_ref = Array{Matrix{Int}}(undef, n_total_patches, n_local_faces)
face_neighbours_3d_ex_2_ref[1, 1] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_2_ref[1, 2] = [2; 2; 1; -1;;]
face_neighbours_3d_ex_2_ref[1, 3] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_2_ref[1, 4] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_2_ref[1, 5] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_2_ref[1, 6] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_2_ref[2, 1] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_2_ref[2, 2] = [1; 2; 1; -1;;]
face_neighbours_3d_ex_2_ref[2, 3] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_2_ref[2, 4] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_2_ref[2, 5] = Matrix{Int64}(undef, 4, 0)
face_neighbours_3d_ex_2_ref[2, 6] = Matrix{Int64}(undef, 4, 0)

# Edge neighbours 
n_total_patches = size(mesh_connectivity_3d_ex_2, 1)
n_local_edges = 12  # hexahedra
edge_neighbours_3d_ex_2_ref = Array{Matrix{Int}}(undef, n_total_patches, n_local_edges)
edge_neighbours_3d_ex_2_ref[1, 1] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[1, 2] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[1, 3] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[1, 4] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[1, 5] = [2; 5; 0; -1;;]
edge_neighbours_3d_ex_2_ref[1, 6] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[1, 7] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[1, 8] = [2; 8; 0; -1;;]
edge_neighbours_3d_ex_2_ref[1, 9] = [2; 12; 0; 1;;]
edge_neighbours_3d_ex_2_ref[1, 10] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[1, 11] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[1, 12] = [2; 9; 0; 1;;]
edge_neighbours_3d_ex_2_ref[2, 1] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[2, 2] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[2, 3] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[2, 4] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[2, 5] = [1; 5; 0; -1;;]
edge_neighbours_3d_ex_2_ref[2, 6] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[2, 7] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[2, 8] = [1; 8; 0; -1;;]
edge_neighbours_3d_ex_2_ref[2, 9] = [1; 12; 0; 1;;]
edge_neighbours_3d_ex_2_ref[2, 10] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[2, 11] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_3d_ex_2_ref[2, 12] = [1; 9; 0; 1;;]

# Vertex neighbours 
n_total_patches = size(mesh_connectivity_3d_ex_2, 1)
n_local_vertices = 8  # hexahedra
vertex_neighbours_3d_ex_2_ref = Array{Matrix{Int}}(undef, n_total_patches, n_local_vertices)
vertex_neighbours_3d_ex_2_ref[1, 1] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_2_ref[1, 2] = [2; 3; 0; 0;;]
vertex_neighbours_3d_ex_2_ref[1, 3] = [2; 2; 0; 0;;]
vertex_neighbours_3d_ex_2_ref[1, 4] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_2_ref[1, 5] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_2_ref[1, 6] = [2; 7; 0; 0;;]
vertex_neighbours_3d_ex_2_ref[1, 7] = [2; 6; 0; 0;;]
vertex_neighbours_3d_ex_2_ref[1, 8] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_2_ref[2, 1] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_2_ref[2, 2] = [1; 3; 0; 0;;]
vertex_neighbours_3d_ex_2_ref[2, 3] = [1; 2; 0; 0;;]
vertex_neighbours_3d_ex_2_ref[2, 4] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_2_ref[2, 5] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_3d_ex_2_ref[2, 6] = [1; 7; 0; 0;;]
vertex_neighbours_3d_ex_2_ref[2, 7] = [1; 6; 0; 0;;]
vertex_neighbours_3d_ex_2_ref[2, 8] = Matrix{Int64}(undef, 4, 0)

# Test MeshTopology
# Compute the mesh topology
mesh_topology_3d_ex_2 = Mantis.Mesh.MeshTopology(mesh_connectivity_3d_ex_2)
# Check if incidence relations are correct 
for i in 1:4
    for j in 1:4
        @test incidence_relation_ex_2_ref[i, j] == mesh_topology_3d_ex_2[i, j]
    end
end

# Test face neighbours
# Compute face neighbours 
face_neighbours_3d_ex_2 = Mantis.Mesh.compute_face_neighbours(mesh_topology_3d_ex_2)
# Check if face neighbours are correct 
for i in 1:n_total_patches
    for j in 1:n_local_faces
        @test face_neighbours_3d_ex_2_ref[i, j] == face_neighbours_3d_ex_2[i, j]
    end
end

# Test edge neighbours
# Compute edge neighbours 
edge_neighbours_3d_ex_2 = Mantis.Mesh.compute_edge_neighbours(mesh_topology_3d_ex_2)
# Check if edge neighbours are correct 
for i in 1:n_total_patches
    for j in 1:n_local_edges
        @test edge_neighbours_3d_ex_2_ref[i, j] == edge_neighbours_3d_ex_2[i, j]
    end
end

# Test vertex neighbours
# Compute vertex neighbours 
vertex_neighbours_3d_ex_2 = Mantis.Mesh.compute_vertex_neighbours(mesh_topology_3d_ex_2)
# Check if edge neighbours are correct 
for i in 1:n_total_patches
    for j in 1:n_local_vertices
        @test vertex_neighbours_3d_ex_2_ref[i, j] == vertex_neighbours_3d_ex_2[i, j]
    end
end

# -----------------------------------------------------------------------------
# Test 2D mesh topology 
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Test mesh 1: 2 patches, configuration 1
# -----------------------------------------------------------------------------
#   1st patch: 1, 2, 3, 4       (vertex indices)
#   2nd patch: 3, 2, 5, 6       (vertex indices)
mesh_connectivity_2d_ex_1 = [[1, 2, 3, 4], [3, 2, 5, 6]]

# Define reference results
# Incidence relations
incidence_relation_2d_ex_1_ref = Matrix{Vector{Vector{Int}}}(undef, 3, 3)
incidence_relation_2d_ex_1_ref[1, 1] = Vector{Int64}[]
incidence_relation_2d_ex_1_ref[1, 2] = [
    [1, 2], [1, 3, 4], [3, 5, 6], [2, 5], [4, 7], [6, 7]
]
incidence_relation_2d_ex_1_ref[1, 3] = [[1], [1, 2], [1, 2], [1], [2], [2]]
incidence_relation_2d_ex_1_ref[2, 1] = [
    [1, 2], [1, 4], [2, 3], [2, 5], [4, 3], [3, 6], [6, 5]
]
incidence_relation_2d_ex_1_ref[2, 2] = Vector{Int64}[]
incidence_relation_2d_ex_1_ref[2, 3] = [[1], [1], [1, 2], [2], [1], [2], [2]]
incidence_relation_2d_ex_1_ref[3, 1] = [[1, 2, 3, 4], [3, 2, 5, 6]]
incidence_relation_2d_ex_1_ref[3, 2] = [[2, 3, 1, 5], [6, 4, -3, 7]]
incidence_relation_2d_ex_1_ref[3, 3] = Vector{Int64}[]

# Edge neighbours 
n_total_patches = size(mesh_connectivity_2d_ex_1, 1)
n_local_edges = 4  # quadrilaterals
edge_neighbours_2d_ex_1_ref = Array{Matrix{Int}}(undef, n_total_patches, n_local_edges)
edge_neighbours_2d_ex_1_ref[1, 1] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_2d_ex_1_ref[1, 2] = [2; 3; 0; -1;;]
edge_neighbours_2d_ex_1_ref[1, 3] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_2d_ex_1_ref[1, 4] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_2d_ex_1_ref[2, 1] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_2d_ex_1_ref[2, 2] = Matrix{Int64}(undef, 4, 0)
edge_neighbours_2d_ex_1_ref[2, 3] = [1; 2; 0; -1;;]
edge_neighbours_2d_ex_1_ref[2, 4] = Matrix{Int64}(undef, 4, 0)

# Vertex neighbours 
n_total_patches = size(mesh_connectivity_2d_ex_1, 1)
n_local_vertices = 4  # quadrilaterals
vertex_neighbours_2d_ex_1_ref = Array{Matrix{Int}}(undef, n_total_patches, n_local_vertices)
vertex_neighbours_2d_ex_1_ref[1, 1] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_2d_ex_1_ref[1, 2] = [2; 2; 0; 0;;]
vertex_neighbours_2d_ex_1_ref[1, 3] = [2; 1; 0; 0;;]
vertex_neighbours_2d_ex_1_ref[1, 4] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_2d_ex_1_ref[2, 1] = [1; 3; 0; 0;;]
vertex_neighbours_2d_ex_1_ref[2, 2] = [1; 2; 0; 0;;]
vertex_neighbours_2d_ex_1_ref[2, 3] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_2d_ex_1_ref[2, 4] = Matrix{Int64}(undef, 4, 0)

# Test MeshTopology
# Compute the mesh topology
mesh_topology_2d_ex_1 = Mantis.Mesh.MeshTopology(mesh_connectivity_2d_ex_1)

# Check if incidence relations are correct 
for i in 1:3
    for j in 1:3
        @test incidence_relation_2d_ex_1_ref[i, j] == mesh_topology_2d_ex_1[i, j]
    end
end

# Test edge neighbours
# Compute edge neighbours 
edge_neighbours_2d_ex_1 = Mantis.Mesh.compute_edge_neighbours(mesh_topology_2d_ex_1)

# Check if edge neighbours are correct 
for i in 1:n_total_patches
    for j in 1:n_local_edges
        @test edge_neighbours_2d_ex_1_ref[i, j] == edge_neighbours_2d_ex_1[i, j]
    end
end

# Test vertex neighbours
# Compute vertex neighbours 
vertex_neighbours_2d_ex_1 = Mantis.Mesh.compute_vertex_neighbours(mesh_topology_2d_ex_1)
# Check if edge neighbours are correct 
for i in 1:n_total_patches
    for j in 1:n_local_vertices
        @test vertex_neighbours_2d_ex_1_ref[i, j] == vertex_neighbours_2d_ex_1[i, j]
    end
end

# -----------------------------------------------------------------------------
# Test 1D mesh topology 
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Test mesh 1: 3 patches, configuration 1
# -----------------------------------------------------------------------------
#   1st patch: 1, 2     (vertex indices)
#   2nd patch: 2, 3     (vertex indices)
#   3rd patch: 3, 4     (vertex indices)
mesh_connectivity_1d_ex_1 = [[1, 2], [2, 3], [3, 4]]

# Define reference results
# Incidence relations
incidence_relation_1d_ex_1_ref = Matrix{Vector{Vector{Int}}}(undef, 2, 2)
incidence_relation_1d_ex_1_ref[1, 1] = Vector{Int64}[]
incidence_relation_1d_ex_1_ref[1, 2] = [[1], [1, 2], [2, 3], [3]]
incidence_relation_1d_ex_1_ref[2, 1] = [[1, 2], [2, 3], [3, 4]]
incidence_relation_1d_ex_1_ref[2, 2] = Vector{Int64}[]

# Vertex neighbours 
n_total_patches = size(mesh_connectivity_1d_ex_1, 1)
n_local_vertices = 2  # lines
vertex_neighbours_1d_ex_1_ref = Array{Matrix{Int}}(undef, n_total_patches, n_local_vertices)
vertex_neighbours_1d_ex_1_ref[1, 1] = Matrix{Int64}(undef, 4, 0)
vertex_neighbours_1d_ex_1_ref[1, 2] = [2; 1; 0; 0;;]
vertex_neighbours_1d_ex_1_ref[2, 1] = [1; 2; 0; 0;;]
vertex_neighbours_1d_ex_1_ref[2, 2] = [3; 1; 0; 0;;]
vertex_neighbours_1d_ex_1_ref[3, 1] = [2; 2; 0; 0;;]
vertex_neighbours_1d_ex_1_ref[3, 2] = Matrix{Int64}(undef, 4, 0)

# Test MeshTopology
# Compute the mesh topology
mesh_topology_1d_ex_1 = Mantis.Mesh.MeshTopology(mesh_connectivity_1d_ex_1)

# Check if incidence relations are correct 
for i in 1:2
    for j in 1:2
        @test incidence_relation_1d_ex_1_ref[i, j] == mesh_topology_1d_ex_1[i, j]
    end
end

# Test vertex neighbours
# Compute vertex neighbours 
vertex_neighbours_1d_ex_1 = Mantis.Mesh.compute_vertex_neighbours(mesh_topology_1d_ex_1)
# Check if edge neighbours are correct 
for i in 1:n_total_patches
    for j in 1:n_local_vertices
        @test vertex_neighbours_1d_ex_1_ref[i, j] == vertex_neighbours_1d_ex_1[i, j]
    end
end

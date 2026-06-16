
module Patches

abstract type AbstractPatch{manifold_dim, incidence_relations_dim} end
abstract type AbstractTensorProductPatch{manifold_dim, incidence_relations_dim} <: AbstractPatch{manifold_dim, incidence_relations_dim} end

struct Line <: AbstractTensorProductPatch{1, 2}
    incidence_relations::NTuple{2, NTuple{2, Vector{Vector{Int}}}}
    n_geometric_objects::NTuple{2, Int}

    function Line()
        incidence_relations = (
            (
                Vector{Vector{Int}}(),             # [1][1]: vertex2vertex (left empty since trivial and not used) 
                [
                    [1],
                    [1]
                ]                                  # [1][2]: vertex2edge
            ),                           
            (
                [
                    [1, 2]
                ],                                 # [2][1]: edge2vertex
                Vector{Vector{Int}}()              # [2][2]: edge2edge (left empty since trivial and not used)
            )                
        )

        n_geometric_objects = (2, 1)

        return new(incidence_relations, n_geometric_objects)
    end
end


struct Quad <: AbstractTensorProductPatch{2, 3}
    incidence_relations::NTuple{3, NTuple{3, Vector{Vector{Int}}}}
    n_geometric_objects::NTuple{3, Int}

    function Quad()
        incidence_relations = (
            (
                Vector{Vector{Int}}(),                # [1][1]: vertex2vertex (left empty since trivial and not used) 
                [
                    [1, 3], 
                    [2, 3], 
                    [2, 4], 
                    [1, 4]
                ],                                    # [1][2]: vertex2edge
                [
                    [1], 
                    [1], 
                    [1], 
                    [1]
                ]                                     # [1][3]: vertex2face
            ),                                    
            (
                [
                    [1, 4], 
                    [2, 3], 
                    [1, 2], 
                    [4, 3]
                ],                                 # [2][1]: edge2vertex
                Vector{Vector{Int}}(),             # [2][2]: edge2edge (left empty since trivial and not used) 
                [
                    [1], 
                    [1], 
                    [1], 
                    [1]
                ]                                  # [2][3]: edge2face
            ),                 
            (
                [
                    [1, 2, 3, 4]
                ],                                 # [3][1]: face2vertex
                [
                    [1, 2, 3, 4]
                ],                                 # [3][2]: face2edge
                Vector{Vector{Int}}()              # [3][3]: face2face  (left empty since trivial and not used) 
            )                 
        )

        n_geometric_objects = (4, 4, 1)

        return new(incidence_relations, n_geometric_objects)
    end
end


struct Hex <: AbstractTensorProductPatch{3, 4}
    incidence_relations::NTuple{4, NTuple{4, Vector{Vector{Int}}}}
    n_geometric_objects::NTuple{4, Int}

    function Hex()
        incidence_relations = (
            (
                Vector{Vector{Int}}(),                      # [1][1]: vertex2vertex (left empty since trivial and not used) 
                [
                    [3, 7, 11], 
                    [3, 8, 12], 
                    [4, 8, 9], 
                    [4, 7, 10], 
                    [2, 6, 11], 
                    [2, 5, 12], 
                    [1, 5, 9], 
                    [1, 6, 10]
                ],                                          # [1][2]: vertex2edge
                [
                    [1, 3, 5], 
                    [2, 3, 5], 
                    [2, 5, 4], 
                    [1, 4, 5], 
                    [1, 3, 6], 
                    [2, 3, 6], 
                    [2, 4, 6], 
                    [1, 4, 6]
                ],                                          # [1][3]: vertex2face
                [
                    [1], 
                    [1], 
                    [1], 
                    [1], 
                    [1], 
                    [1], 
                    [1], 
                    [1]
                ]                                           # [1][4]: vertex2volume
            ),                                     
            (
                [
                    [8, 7], 
                    [5, 6], 
                    [1, 2], 
                    [4, 3], 
                    [6, 7], 
                    [5, 8], 
                    [1, 4], 
                    [2, 3], 
                    [3, 7], 
                    [4, 8], 
                    [1, 5], 
                    [2, 6]
                ],                                     # [2][1]: edge2vertex
                Vector{Vector{Int}}(),                 # [2][2]: edge2edge (left empty since trivial and not used) 
                [
                    [4, 6],
                    [3, 6],
                    [3, 5],
                    [4, 5],
                    [2, 6],
                    [1, 6],
                    [1, 5],
                    [2, 5],
                    [2, 4],
                    [1, 4],
                    [1, 3],
                    [2, 3]
                ],                                     # [2][3]: edge2face
                [
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1]
                ]                                     # [2][4]: edge2volume
            ),                                         
            (
                [
                    [1, 5, 8, 4], 
                    [2, 3, 7, 6],
                    [1, 2, 6, 5],
                    [4, 8, 7, 3],
                    [1, 4, 3, 2],
                    [5, 6, 7, 8]
                ],                                     # [3][1]: face2vertex
                [
                    [11, 10, 7, 6],
                    [12, 9, 8, 5],
                    [11, 12, 3, 2],
                    [10, 9, 4, 1],
                    [7, 8, 3, 4],
                    [6, 5, 2, 1]
                ],                                     # [3][2]: face2edge
                Vector{Vector{Int}}(),                 # [3][3]: face2face  (left empty since trivial and not used)
                [
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1]
                ]                                       # [3][4]: face2volume
            ),                                        
            (
                [
                    [1, 2, 3, 4, 5, 6, 7, 8]
                ],                                      # [4][1]: volume2vertex
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                ],                                      # [4][2]: volume2edge
                [
                    [1, 2, 3, 4, 5, 6]
                ],                                      # [4][3]: volume2face
                Vector{Vector{Int}}()                   # [4][4]: volume2volume (left empty since trivial and not used)
            )
        )                 

        n_geometric_objects = (8, 12, 6, 1)

        return new(incidence_relations, n_geometric_objects)
    end
end

# # Indexing.
# i and k and geometric dimension indices, i.e., if geometric dimension is n
# then the index is (n + 1), this is because julia starts indices at 1 and vertices
# have dimension 0.
Base.getindex(element::AbstractTensorProductPatch, i::Int, k::Int) = element.incidence_relations[i][k]

# Sizes.
# Provide quick access to the number of local geometric objects in each dimension
# (vertices, edges) in 1D
# (vertices, edges, surfaces) in 2D
# (vertices, edges, surfaces, volumes) in 3D
Base.size(element::AbstractTensorProductPatch) = element.n_geometric_objects
# geometric_dim_id is the index associated to the geometric dimension. Geometric dimension n
# has index (n + 1), this is done to keep consistency with julia indices that start at 1 and
# not at 0 (vertices have geometric dimension 0).
Base.size(element::AbstractTensorProductPatch, geometric_dim_id::Int) =
    element.n_geometric_objects[geometric_dim_id]

end
struct ParentChildRelations{PCE, CPE, PCB, CPB}
    parent_to_children_elements::PCE
    child_to_parent_elements::CPE
    parent_to_children_basis::PCB
    child_to_parents_basis::CPB
end

function get_parent_to_children_elements(relations::ParentChildRelations)
    return relations.parent_to_children_elements
end

function get_child_to_parent_elements(relations::ParentChildRelations)
    return relations.child_to_parent_elements
end

function get_parent_to_children_basis(relations::ParentChildRelations)
    return relations.parent_to_children_basis
end

function get_child_to_parents_basis(relations::ParentChildRelations)
    return relations.child_to_parents_basis
end

function get_element_children(
    relations::ParentChildRelations{PCE, CPE, PCB, CPB}, parent::Int
) where {V <: AbstractVector{Int}, PCE <: Vector{V}, CPE, PCB, CPB}
    return get_parent_to_children_elements(relations)[parent]
end

function get_element_children(
    relations::ParentChildRelations{PCE, CPE, PCB, CPB}, parent::Int
) where {PCE <: Function, CPE, PCB, CPB}
    return get_parent_to_children_elements(relations)(parent)
end

function get_element_parent(
    relations::ParentChildRelations{PCE, CPE, PCB, CPB}, child::Int
) where {PCE, CPE <: AbstractVector{Int}, PCB, CPB}
    return get_child_to_parent_elements(relations)[child]
end

function get_element_parent(
    relations::ParentChildRelations{PCE, CPE, PCB, CPB}, child::Int
) where {PCE, CPE <: Function, PCB, CPB}
    return get_child_to_parent_elements(relations)(child)
end

function get_basis_children(
    relations::ParentChildRelations{PCE, CPE, PCB, CPB}, parent::Int
) where {PCE, CPE, V <: AbstractVector{Int}, PCB <: Vector{V}, CPB}
    return get_parent_to_children_basis(relations)[parent]
end

function get_basis_children(
    relations::ParentChildRelations{PCE, CPE, PCB, CPB}, parent::Int
) where {PCE, CPE, PCB <: Function, CPB}
    return get_parent_to_children_basis(relations)(parent)
end

function get_basis_parents(
    relations::ParentChildRelations{PCE, CPE, PCB, CPB}, child::Int
) where {V <: AbstractVector{Int}, PCE, CPE, PCB, CPB <: Vector{V}}
    return get_child_to_parents_basis(relations)[child]
end

function get_basis_parents(
    relations::ParentChildRelations{PCE, CPE, PCB, CPB}, child::Int
) where {PCE, CPE, PCB, CPB <: Function}
    return get_child_to_parents_basis(relations)(child)
end

# get_parent_to_child_relations(TS::TensorProductTwoScaleOperator) =
#     parent -> get_element_children(TS, parent)

# get_child_to_parent_relations(TS::TensorProductTwoScaleOperator) =
#     child -> get_element_parent(TS, child)

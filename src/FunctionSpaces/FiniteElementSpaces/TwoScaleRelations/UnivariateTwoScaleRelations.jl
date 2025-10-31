function get_element_children(parent_element_id::Int, num_subdivisions::Int)
    offset = (parent_element_id - 1) * num_subdivisions

    return (offset + 1):(offset + num_subdivisions)
end

function get_element_parent(child_element_id::Int, num_subdivisions::Int)
    return floor(Int, (child_element_id - 1) / num_subdivisions + 1)
end

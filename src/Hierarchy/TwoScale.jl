"""
Functions for 2-scale relations.

"""

# Getters for elements

function element_ranges_to_tuple_list(finer_ranges::Vector{Vector{Int}}, subdivisions::NTuple{n, Int}) where {n}
    finer_elements = Vector{NTuple{n, Int}}(undef, prod(subdivisions))

    el_count = 1
    @inbounds for el_id in Iterators.product(finer_ranges...)
        finer_elements[el_count] = el_id
        el_count += 1
    end
    
    return finer_elements
end

function get_finer_elements(element_id::Int, subdivisions::Int)
    return collect((1:subdivisions) .+ (element_id-1)*subdivisions) 
end

function get_finer_elements(element_id::NTuple{n, Int}, subdivisions::NTuple{n, Int}) where {n}
    finer_ranges = Vector{Vector{Int}}(undef, n)

    for i in 1:1:n
        finer_ranges[i] = get_finer_elements(element_id[i], subdivisions[i])
    end

    return element_ranges_to_tuple_list(finer_ranges, subdivisions)
end
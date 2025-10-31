function build_two_scale_operator(
    parent_space::DirectSumSpace{manifold_dim, num_components},
    num_subdivisions::NTuple{manifold_dim, Int},
) where {manifold_dim, num_components}
    # build two scale operators for all component spaces
    comp_parent_spaces = get_component_spaces(parent_space)
    twoscale_data = ntuple(
        component ->
            build_two_scale_operator(comp_parent_spaces[component], num_subdivisions),
        num_components,
    )
    comp_twoscale_operators = ntuple(
        component -> twoscale_data[component][1], num_components
    )
    comp_child_spaces = ntuple(component -> twoscale_data[component][2], num_components)
    child_space = DirectSumSpace(comp_child_spaces)

    return TwoScaleOperator(
        parent_space,
        child_space,
        SparseArrays.blockdiag(
            ntuple(
                component -> get_global_subdiv_matrix(comp_twoscale_operators[component]),
                num_components,
            )...,
        ),
        get_parent_to_children_elements(comp_twoscale_operators[1]),
        get_child_to_parent_elements(comp_twoscale_operators[1]),
    ),
    child_space
end

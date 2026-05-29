function space_evaluate(space::Bernstein, ::Int, der_key::Int)
    # The second argument is ignored, since this basis is element agnostic.
    p = get_polynomial_degree(space)

    return (point, basis, _) -> _dbpoly(p, basis - 1, first(der_key), point[1])
end

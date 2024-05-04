@doc raw"""
Tensor product geometry of two geometries.

An ``(n, m)`` geometry $\Phi$ is a collection of ``L`` mappings ``\left\{\Phi_{i}\right\}_{i=1}^{L}`` 
that map the canonical ``n``-dimensional domain, ``\Omega^{0} := [0, 1]^{n}`` into ``L`` 
``n``-dimensional simply connected subdomains, ``\Omega^{1}_{i}`` with ``i = 1, \dots, L``, of ``\mathbb{R}^{m}``.
Moreover, ``\bigcap_{i=1}^{L}\Omega^{1}_{i} = \emptyset`` and 
``\overline{\Omega}^{1}_{i} \cap \overline{\Omega}^{1}_{j} \subset \partial\Omega^{1}_{i} \cup \partial\Omega^{1}_{j}``
with ``i,j = 1, \dots, L``.

Note that 

```math
\Phi_{i}(\xi_{1}, \dots, \xi_{n}) = (x_{1}, \dots, x_{m}), 
```

and we use ``\Phi_{i, j} = x_{j}`` to mean the ``j``-th component of the mapping ``\Phi`` of element ``i``.

Given an ``(n_{1}, m_{1})`` geometry ``\Phi^{1}`` of ``L_{1}`` mappings and an ``(n_{2}, m_{2})`` geometry ``\Phi^{2}``
of ``L_{2}`` mappings, i.e.,

```math
\Phi^{1}_{i}: [0, 1]^{n_{1}} \mapsto \Omega^{1}_{i} \subset \mathbb{R}^{m_{1}}, \quad i = 1, \dots, L_{1}
```

and

```math
\Phi^{2}_{i}: [0, 1]^{n_{2}} \mapsto \Omega^{2}_{i} \subset \mathbb{R}^{m_{2}}, \quad i = 1, \dots, L_{2}
```

the tensor product geometry ``\Phi := \Phi^{1}\otimes\Phi^{2}`` is an ``(n_{1} + n_{2}, m_{1} + m_{2})`` geometry
made up of a collection of ``L_{1}L_{2}`` mappings ``\Phi_{k}``

```math
\Phi_{k = L_1(j - 1) + I}: [0, 1]^{n_{1}} \times [0, 1]^{n_{2}} \mapsto \Omega_{k} = \Omega^{1}_{i}\times\Omega^{2}_{j} \subset \mathbb{R}^{m_{1} + m_{2}}, \quad i = 1, \dots, L_{1}, \text{ and } j = 1, \dots, L{2}.
```

Specifically, we have

```math
\Phi_{L_1(j - 1) + i, l}(\xi_{1}, \dots, \xi_{n}) := \left\{ 
\begin{array}{ll}
\Phi^{1}_{i, l}(\xi_{1}, \dots, \xi_{n_{1}}), & \quad \text{if } l \leq n_{1}\\
\Phi^{2}_{j, l - n_{1}}(\xi_{n_{1} + 1}, \dots, \xi_{n_{1} + n_{2}}), & \quad \text{if } n_{1} < l \leq n_{1} + n_{2} 
\end{array}
\right.\,,
\quad i = 1, \dots, L_{1},\quad j = 1, \dots, L_{2}, \text{ and } l = 1, \dots, m_{1} + m_{2}\,.

```

The Jacobian of this geometry 

```math
J^{k}_{l,v} := \frac{\partial \Phi_{k, l}}{\partial \xi_{v}} 
```

is given by 

```math
\frac{\partial\Phi_{L_1(j - 1) + i, l}}{\partial\xi_{v}}(\xi_{1}, \dots, \xi_{n}) := \left\{ 
\begin{array}{ll}
\frac{\partial\Phi^{1}_{i, l}}{\partial \xi_{v}}(\xi_{1}, \dots, \xi_{n_{1}}), & \quad \text{if } l \leq n_{1}, \text{  and  } v \leq m_{1}\\
\frac{\partial\Phi^{2}_{j, l - n_{1}}}{\partial \xi_{u - m_{1}}}(\xi_{n_{1} + 1}, \dots, \xi_{n_{1} + n_{2}}), & \quad \text{if } n_{1} < l \leq n_{1} + n_{2}, \text{  and  } m_{1} < v \leq m_{1} + m_{2} \\
0 & \quad\text{otherwise}
\end{array}
\right.\,,
\quad i = 1, \dots, L_{1}, \quad j = 1, \dots, L_{2}, \text{ and } l = 1, \dots, m_{1} + m_{2}\,.

```

"""
struct TensorProductGeometry{n, m} <: AbstractGeometry{n, m}
    geometry_1::AbstractGeometry
    geometry_2::AbstractGeometry
    domain_dims::NTuple{2, Int}
    image_dims::NTuple{2, Int}
    n_elements_per_geometry::NTuple{2, Int}
    n_elements::Int
    cartesian_indices::CartesianIndices
    
    function TensorProductGeometry(geometry_1::AbstractGeometry{n_1, m_1}, geometry_2::AbstractGeometry{n_2, m_2}) where {n_1, m_1, n_2, m_2}
        n_elements_per_geometry = (get_num_elements(geometry_1), get_num_elements(geometry_2))
        n_elements = prod(n_elements_per_geometry)
        domain_dims = (get_domain_dim(geometry_1), get_domain_dim(geometry_2))
        image_dims = (get_image_dim(geometry_1), get_image_dim(geometry_2))
        n = sum(domain_dims)
        m = sum(image_dims) 
        cartesian_indices = CartesianIndices(n_elements_per_geometry)
        return new{n, m}(geometry_1, geometry_2, domain_dims, image_dims, n_elements_per_geometry, n_elements, cartesian_indices)
    end
end

function get_num_elements(geometry::TensorProductGeometry{n, m}) where {n, m}
    return geometry.n_elements
end

function get_domain_dim(::TensorProductGeometry{n, m}) where {n, m}
    return n
end

function get_image_dim(::TensorProductGeometry{n, m}) where {n, m}
    return m
end

@doc raw"""
    evaluate(geometry::TensorProductGeometry{n, m}, element_idx::Int, ξ::NTuple{n, Vector{Float64}})

Evaluates the tensor product geometry `geometry` at element `element_idx` and at the tensor product of the 
`NTuple` coordinates `ξ`.

In other words, given the `NTuple` `ξ` of ``n`` `Vectors`, ``\boldsymbol{\xi}^{i}``, ``i=1, \dots, n``,
each containing ``m_{i}`` unidimensional coordinates ``\xi^{i}_{j}``, ``i = 1, \dots, n`` and `` j = 1, \dots m_{i}``,
evaluates the tensor product geometry at the element `element_idx` and at the ``\prod_{i=1}^{n}m_{i}`` tensor product
points ``V_{k = j_{1} + \sum_{i=2}^{n} (j_{i} - 1)\prod_{l=1}^{i-1}m_{l}} = (\xi^{1}_{j_{1}}, \dots, \xi^{n}_{j_{n}})``, 
with ``j_{i} = 1, \dots, m_{i}``.

The output is a matrix, ``\boldsymbol{\mathsf{X}}`` of dimensions ``\left(\prod_{i=1}^{n}m_{i}\right) \times m`` 
(the number of tensor product points where the geometry is evaluated in element `element_idx`, and the dimension of the
embedding space to where the canonical element is mapped into. Specifically:

```math
\boldsymbol{\mathsf{X}}_{k, l} = \Phi_{r, l}(\xi^{1}_{j_{1}}, \dots, \xi^{n}_{j_{n}}),
```

where ``r = \mathtt{element\_idx}``, and ``k = j_{1} + \sum_{i=2}^{n} (j_{i} - 1)\prod_{l=1}^{i-1}m_{l}``, as before.

# Arguments
- `geometry::TensorProductGeometry{n, m}`: tensor product geometry of elements of dimension `n` embedded in a domain of 
   dimension `m`.
- `element_idx::Int`: the index of the element of the tensor product geometry where to evaluate the geometry.
- `ξ::NTuple{n,Vector{Float64}}`: the ``n`` unidimensional vectors with ``m_{i}``, ``i = 1, \dots, n``, 
   coordinates where to evaluate the geometry in the canonical domain `[0, 1]^{n}`. 
   The geometry will be evaluated at the `M` points, ``\mathtt{M} = \prod_{i=1}^{n}m_{i}``, generated by 
   the tensor product of the unidimensional coordinates, as explained above.

# Returns 
`x::Array{Float64, 2}(M, m)`: the evaluation of the geometry at the element `element_idx`, and the tensor product points
 of the unidimensional coordinates `\xi`. See above for more details.

"""
function evaluate(geometry::TensorProductGeometry{n, m}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, m}
    element_geometries_idx = geometry.cartesian_indices[element_idx]
    
    n_points = prod(size.(ξ, 1))  # the total number of points to evaluate, it is a tensor product of the coordinates to sample in each direction
    x = zeros(Float64, n_points, m)

    x[:, 1:geometry.image_dims[1]] .= evaluate(geometry.geometry_1, element_geometries_idx[1], ξ[1:geometry.domain_dims[1]])
    x[:, (geometry.image_dims[1]+1):end] .= evaluate(geometry.geometry_2, element_geometries_idx[2], ξ[(geometry.domain_dims[1]+1):end])
    
    return x
end

function evaluate(geometry::TensorProductGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
    return evaluate(geometry, element_idx, ntuple(i -> [ξ[i]], n))[1, :]
end

@doc raw"""
    jacobian(geometry::TensorProductGeometry{n, m}, element_idx::Int, ξ::NTuple{n,Vector{Float64}})

Evaluates the Jacobian of the tensor product geometry `geometry` at element `element_idx` and at the tensor product of the 
`NTuple` coordinates `ξ`.

In other words, given the `NTuple` `ξ` of ``n`` `Vectors`, ``\boldsymbol{\xi}^{i}``, ``i=1, \dots, n``,
each containing ``m_{i}`` unidimensional coordinates ``\xi^{i}_{j}``, ``i = 1, \dots, n`` and `` j = 1, \dots m_{i}``,
evaluates the Jacobian of the tensor product geometry at the element `element_idx` and at the ``\prod_{i=1}^{n}m_{i}`` tensor product
points ``V_{k = j_{1} + \sum_{i=2}^{n} (j_{i} - 1)\prod_{l=1}^{i-1}m_{l}} = (\xi^{1}_{j_{1}}, \dots, \xi^{n}_{j_{n}})``, 
with ``j_{i} = 1, \dots, m_{i}``.

The output is a matrix, ``\boldsymbol{\mathsf{J}}`` of dimensions ``\left(\prod_{i=1}^{n}m_{i}\right) \times m \times n`` 
(the number of tensor product points where the geometry is evaluated in element `element_idx`, the dimension of the
embedding space to where the canonical element is mapped into, and the dimension of the canonical element, which is the
same as the dimension of the element's manifold). Specifically:

```math
\boldsymbol{\mathsf{J}}_{k, l, s} = \frac{\partial\Phi_{r, l}}{\partial\xi_{s}}(\xi^{1}_{j_{1}}, \dots, \xi^{n}_{j_{n}}),
```

where ``r = \mathtt{element\_idx}``, and ``k = j_{1} + \sum_{i=2}^{n} (j_{i} - 1)\prod_{l=1}^{i-1}m_{l}``, as before.

# Arguments
- `geometry::TensorProductGeometry{n, m}`: tensor product geometry of elements of dimension `n` embedded in a domain of 
   dimension `m`.
- `element_idx::Int`: the index of the element of the tensor product geometry where to evaluate the geometry.
- `ξ::NTuple{n,Vector{Float64}}`: the ``n`` unidimensional vectors with ``m_{i}``, ``i = 1, \dots, n``, 
   coordinates where to evaluate the geometry in the canonical domain `[0, 1]^{n}`. 
   The geometry will be evaluated at the `M` points, ``\mathtt{M} = \prod_{i=1}^{n}m_{i}``, generated by 
   the tensor product of the unidimensional coordinates, as explained above.

# Returns 
`J::Array{Float64, 2}(M, m)`: the evaluation of the Jacobian of the geometry at the element `element_idx`, 
 and the tensor product points of the unidimensional coordinates `\xi`. See above for more details.

"""
function jacobian(geometry::TensorProductGeometry{n, m}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, m}
    element_geometries_idx = geometry.cartesian_indices[element_idx]

    n_points = prod(size.(ξ, 1))  # the total number of points to evaluate, it is a tensor product of the coordinates to sample in each direction
    J = zeros(Float64, n_points, m, n)

    J[:, 1:geometry.image_dims[1], 1:geometry.domain_dims[1]] .= jacobian(geometry.geometry_1, element_geometries_idx[1], ξ[1:geometry.domain_dims[1]])  # the Jacobian contribution from geometry 1
    J[:, (geometry.image_dims[1]+1):end, (geometry.domain_dims[1]+1):end] .= jacobian(geometry.geometry_2, element_geometries_idx[2], ξ[(geometry.domain_dims[1]+1):end])  # the Jacobian contribution from geometry 2
    
    return J
end

function jacobian(geometry::TensorProductGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
    return jacobian(geometry, element_idx, ntuple(i -> [ξ[i]], n))
end

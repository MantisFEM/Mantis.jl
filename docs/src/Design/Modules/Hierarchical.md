```@meta
CurrentModule = Mantis.Hierarchical
```

# Hierarchical

Adaptive methods rely on hierarchical constructions: a sequence of levels, each
finer than the previous one, where only some objects of each level are *active*.
The `Hierarchical` module provides the building blocks for such constructions,
independently of what the objects are. The same tools are used for hierarchies
of elements, in [`Mantis.Geometry.HierarchicalGeometry`](@ref), and for
hierarchies of basis functions, in
[`Mantis.FunctionSpaces.HierarchicalSpace`](@ref), for example.

```@docs
Hierarchical
```

## [Core concepts](@id HierarchicalConcepts)

A hierarchy is built from three ingredients, each with its own section below.

1. **Sets** on each level, such as a geometry or a function space. A finer set
   is obtained from a coarser one through a
   [refinement](@ref HierarchicalRefinement).
2. **Parent-child relations** between the objects of consecutive levels are
   provided by a [scaling](@ref HierarchicalScalings).
3. **Active objects** on each level, stored in an [`ActiveInfo`](@ref).

Consider a 1D geometry with 4 elements on the first level, uniformly refined into 2 elements
each. Some elements are refined (inactive) and replaced by their children:

```
x inactive
- active

Level 1: |---1---|---2---|xxxxxxx|---4---|
Level 2: |xxx|xxx|xxx|xxx|-5-|-6-|xxx|xxx|
```
There are two ways active objects, in this case elements, are numbered in
`Mantis`. The *level-wise* numbering is the numbering of the objects within the
set of their level: the active elements above are elements `1`, `2` and `4` of
level 1, and elements `5` and `6` of level 2. The *hierarchical* numbering is
obtained by concatenating the active objects of all levels: the active elements
above are elements `1` to `5`. A hierarchical construction is indexed with the
hierarchical numbering, while the level-wise numbering is used to evaluate the
objects on their level.

## [Refinement](@id HierarchicalRefinement)

A refinement holds a parent set and a method to produce a child set from it. The
child set is created lazily, by calling the refinement with no arguments. This
way a refinement can be stored without instantiating the child set, which is
useful since in adaptive methods it is not known in advance if a refinement will
happen.

```@docs
AbstractRefinement
RefinementMethod
RefinementExplicit
Refinement
```

The parent set and the refinement method of a refinement can be retrieved with
the following getters.

```@docs
get_parent(::AbstractRefinement)
get_method
```

## [Scalings](@id HierarchicalScalings)

A scaling relates a parent set to a child set, and allows traversing between
parent and child objects.

### [Relations](@id HierarchicalRelations)

The connectivity between parent and child objects is described by `Relations`,
which store a relation in each direction: from parents to children
([`PC`](@ref)) and from children to parents ([`CP`](@ref)).

```@docs
RelationType
PC
CP
RelationMethod
RelationExplicit
Relations
get_parent_to_children
get_child_to_parents
```

### [Scaling types](@id HierarchicalScalingTypes)

`Mantis` has two concrete scalings. A `Scaling` only provides the relations
between parent and child objects, which is enough for geometries. A
`MatrixScaling` additionally stores a scaling matrix, which expresses parent
objects as linear combinations of child objects, as needed for function spaces.
In this case, the relations are given by the sparsity pattern of the scaling
matrix.

```@docs
AbstractScaling
Scaling
MatrixScaling
```

The sets and relations of any scaling can be inspected with the following
methods.

```@docs
get_parent(::AbstractScaling)
get_child
get_children
get_parents
get_relations
get_sets
```

The scaling matrix of a `MatrixScaling` can be accessed with the following
methods.

```@docs
get_scaling_matrix
view_scaling_matrix
```

::: details Internals: building scaling matrices

We explain how scaling matrices are built. However, this is considered an
implementation detail.

The constructors of `MatrixScaling` build the scaling matrix with the following
function, which calls the provided scaling method on the parent and child sets.

```@docs
build_scaling_matrix
```

:::

## [Active objects](@id HierarchicalActiveInfo)

The active objects of each level are stored in an `ActiveInfo`. Next to the
level-wise ids of the active objects, it stores lookup tables to efficiently
convert between the level-wise and hierarchical numbering (see [Core
concepts](@ref HierarchicalConcepts)).

```@docs
ActiveInfo
```

The active objects can be inspected with the following getters.

```@docs
get_level_ids
get_level_set
get_level_sets
get_level_num_ids
get_num_levels
get_num_objects
get_level
```

The conversion between the hierarchical and level-wise numbering is done with
the following methods.

```@docs
convert_to_level_id
convert_to_level_and_level_id
convert_to_hier_id
```

The active objects can be changed with `update`, which removes objects from a
level and adds objects to the next one. `update` never mutates its input, but
returns an updated copy instead.

```@docs
update
```

::: details Internals: lookup tables

We explain how the lookup tables of an `ActiveInfo` are stored. However, this is
considered an implementation detail.

The following getters give access to the lookup tables used to convert between
numberings, and the following methods append an empty level. You should not need
these functions unless you are extending the functionality of `ActiveInfo`.

```@docs
get_level_lookup
get_level_cum_num_ids
add_level!
add_level
_convert_to_level_id
```

:::

## [Hierarchies](@id HierarchicalHierarchies)

A hierarchy combines the previous ingredients: a sequence of scalings, where the
child set of each scaling is the parent set of the next one, together with an
`ActiveInfo` with the active objects of each level.

```@docs
AbstractHierarchy
Hierarchy
TreeHierarchy
```
A `Hierarchy` is the general hierarchy, used, for example, for the active basis
functions of a hierarchical space. A `TreeHierarchy` is meant for objects with a
single parent, where it is not meaningful to have both a parent and its children
active. This is the case for the elements of a hierarchical geometry.

All hierarchies can be inspected with the following getters. Note that a
hierarchy also supports the getters of [Active
objects](@ref HierarchicalActiveInfo), such as [`convert_to_hier_id`](@ref).

```@docs
get_scalings
get_scaling
get_active_info
get_set
get_tree_ids
get_descendants
```

A hierarchy is refined by deactivating objects on a level and activating their
children on the next level. If the finest level is refined, a new level is
created using the provided scaling.

```@docs
refine
```

::: warning Refining is type unstable

The number of levels is a type parameter of a hierarchy. Since refining the
finest level adds a level, the output type of [`refine`](@ref) and
[`update`](@ref) depends on runtime values. Call such methods behind a function
barrier, so that the instability does not propagate.

:::

::: details Internals: hierarchy construction

We explain some internals related to constructing and updating hierarchies.
However, this is considered an implementation detail.

When a hierarchy is constructed, the following functions check that consecutive
scalings form a type-stable hierarchy and, for a `TreeHierarchy`, that no active
object has active descendants. They also compute the tree ids returned by
[`get_tree_ids`](@ref).

```@docs
_warn_stability
_check_tree
_tree_ids
```

When a hierarchy is updated, the following functions append a scaling for a
newly created level.

```@docs
add_scaling
_add_scaling
```

:::

## [Example](@id HierarchicalExample)

We first build a hierarchical geometry, and then a hierarchical space on top of
it.

### [HierarchicalGeometry](@id HierarchicalExampleGeometry)

We build the hierarchy from [Core concepts](@ref HierarchicalConcepts). We start
from a 1D geometry with 4 elements, and a uniform scaling that subdivides each
element in 2.
```@example HierarchicalExample
using Mantis
using Mantis.Hierarchical

geo_l1 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5),))
scaling = Geometry.scaling_uniform(geo_l1, 2)
nothing # hide
```
The scaling relates each element of level 1 to its children on level 2. For
example, element `2` of level 1 is subdivided into elements `3` and `4` of level
2,
```@example HierarchicalExample
get_children(scaling, 2)
```
and, conversely, element `3` of level 2 has element `2` of level 1 as parent.
```@example HierarchicalExample
get_parents(scaling, 3)
```

The active elements are elements `1`, `2` and `4` of level 1, and elements `5`
and `6` of level 2, which gives 5 active elements in total.
```@example HierarchicalExample
active_info = ActiveInfo([[1, 2, 4], [5, 6]])
get_num_objects(active_info)
```
The fourth active element is element `5` of level 2,
```@example HierarchicalExample
convert_to_level_and_level_id(active_info, 4)
```
while element `6` of level 2 is the fifth active element.
```@example HierarchicalExample
convert_to_hier_id(active_info, 2, 6)
```

Since each element has a single parent, the elements form a `TreeHierarchy`. Its
tree ids contain, for each level, the active elements and the refined elements
with active children.
```@example HierarchicalExample
hierarchy = TreeHierarchy(active_info, scaling)
get_tree_ids(hierarchy)
```

Refining element `5` of level 2 creates a third level, so a scaling from level 2
to level 3 has to be provided. Element `5` is then replaced by its children,
elements `9` and `10` of level 3.
```@example HierarchicalExample
next_scaling = Geometry.scaling_uniform(get_child(scaling), 2)
refined = refine(hierarchy, 2, [5], next_scaling)
get_level_ids(refined)
```

The refined hierarchy defines a
[`Mantis.Geometry.HierarchicalGeometry`](@ref), whose elements are the active
elements of the hierarchy.
```@example HierarchicalExample
hgeo = Geometry.HierarchicalGeometry(refined)
Geometry.get_num_elements(hgeo)
```

Finally, the sets of each level do not need to exist in advance: they can be
created lazily with a `Refinement`. The child geometry is only created when the
refinement is called, and has twice as many elements as its parent.
```@example HierarchicalExample
refinement = Refinement(geo_l1, geo -> Geometry.refinement_uniform(geo, 2))
Geometry.get_num_elements(refinement())
```

### [HierarchicalSpace](@id HierarchicalExampleSpace)

We now build a [`Mantis.FunctionSpaces.HierarchicalSpace`](@ref) on the
hierarchical geometry from the previous section. On the first level, we use
linear B-splines with ``C^0`` regularity. Each finer level is obtained by
subdividing the elements in 2 and elevating the degree by 1, both through lazy
`Refinement`. From level 1 to 2, we subdivide and elevate in two separate
refinements, while from level 2 to 3, we use a single refinement that composes
both.
```@example HierarchicalExample
subdivide(space) = FunctionSpaces.refinement_uniform(space, 2)
elevate(space) = FunctionSpaces.refinement_degree(space, 1)
subdivide_and_elevate = elevate ∘ subdivide

bsp_l1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (4,), (1,), (0,))
bsp_l12 = Refinement(bsp_l1, subdivide)()
bsp_l2 = Refinement(bsp_l12, elevate)()
bsp_l3 = Refinement(bsp_l2, subdivide_and_elevate)()
map(FunctionSpaces.get_polynomial_degree, (bsp_l1, bsp_l2, bsp_l3))
```
Note that the parametric geometry of the space on each level has to be the
geometry on the same level of the hierarchical geometry. This is the case here,
since the spaces are subdivided in the same way as the geometries.

Each `MatrixScaling` composes the scaling matrices of the subdivision and the
degree elevation, relating the functions of a level directly to those of the
next level. From level 1 to 2, the scaling is given all the intermediate spaces
and composes the two scaling matrices itself. From level 2 to 3, there is no
intermediate space, so a single scaling method builds it and multiplies both
matrices.
```@example HierarchicalExample
elevate_matrix(parent, child) = FunctionSpaces.scaling_matrix_degree(parent, child, 1)
function subdivide_and_elevate_matrix(parent, child)
    subdivided = subdivide(parent)
    subdivide_matrix = FunctionSpaces.scaling_matrix_uniform(parent, subdivided)

    return elevate_matrix(subdivided, child) * subdivide_matrix
end

scaling_1 = MatrixScaling(
    (bsp_l1, bsp_l12, bsp_l2), FunctionSpaces.scaling_matrix_uniform, elevate_matrix
)
scaling_2 = MatrixScaling(bsp_l2, bsp_l3, subdivide_and_elevate_matrix)
nothing # hide
```
::: info Both ways are equivalent

Refining and scaling in separate steps, as from level 1 to 2, or with a single
composed function, as from level 2 to 3, gives the same spaces and scaling
matrices. Separate steps reuse the existing refinement and scaling methods
directly, while a composed function is convenient when the intermediate spaces
are not needed.

:::

For example, the second function of level 1 is a combination of the following
functions of level 2.
```@example HierarchicalExample
get_children(scaling_1, 2)
```

We can now build the hierarchical space with both hierarchical B-splines,
[`Mantis.FunctionSpaces.HB`](@ref), and truncated hierarchical B-splines,
[`Mantis.FunctionSpaces.THB`](@ref). Their active basis functions are the same,
but on element `5` (element `9` of level 3), the hierarchical B-splines include
the coarser functions `2` and `6`,
```@example HierarchicalExample
scalings = (scaling_1, scaling_2)
hb = FunctionSpaces.HierarchicalSpace(
    hgeo, scalings, FunctionSpaces.SelectionStandard, FunctionSpaces.HB
)
thb = FunctionSpaces.HierarchicalSpace(
    hgeo, scalings, FunctionSpaces.SelectionStandard, FunctionSpaces.THB
)
sort(FunctionSpaces.get_basis_indices(hb, 5))
```
while these functions are truncated for the truncated hierarchical B-splines.
```@example HierarchicalExample
sort(FunctionSpaces.get_basis_indices(thb, 5))
```
As a result, only the truncated hierarchical B-splines form a partition of
unity.
```@example HierarchicalExample
function is_partition_of_unity(space)
    xi = Points.PointSet((LinRange(0.0, 1.0, 5),))
    for element_id in 1:FunctionSpaces.get_num_elements(space)
        # Values of the basis functions on the element, one column per function.
        values = FunctionSpaces.evaluate(space, element_id, xi)[1][1][1][1]
        if !all(isapprox(1.0), sum(values; dims=2))
            return false
        end
    end

    return true
end

is_partition_of_unity(hb), is_partition_of_unity(thb)
```
The difference is clearly visible when plotting the basis functions. With
hierarchical B-splines, the coarse linear functions span the refined region in
full,
```@example HierarchicalExample
using CairoMakie #hide
using DisplayAs #hide
DisplayAs.Text(DisplayAs.PNG(Mantis.Plot.plot_basis(hb))) #hide
```
while with truncated hierarchical B-splines, they are truncated in the refined
region.
```@example HierarchicalExample
DisplayAs.Text(DisplayAs.PNG(Mantis.Plot.plot_basis(thb))) #hide
```

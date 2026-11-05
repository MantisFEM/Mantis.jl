```@meta
CurrentModule = Mantis.Topology
```

# Topology

```@docs
Topology
```

# [Topology Building Blocks](@id TopologyBuildingBlocks)

## Abstract Types

```@docs
AbstractTopology
AbstractPatch
AbstractTensorProductPatch
```

## [Available Patch Types](@id TopologyPatchTypes)
Every [`AbstractTopology`](@ref) is build using patches of a given type. 
These are all subtypes of [`AbstractPatch`](@ref).
Currently, the following patch types are available:

```@docs
LINE
QUAD
HEX
```

The above constants are the objects that you should use when creating a new mesh.
Since each patch is of the same type, all patch-related information is the same for all of 
them. 
By using the constant instance, all this information only needs to be created once.
The non-constant versions are shown below for completeness.

```@docs
Line
Quad
Hex
```

You can always choose yourself which patch type you would like to use. 
However, if you want to use a tensor product patch of a given manifold dimension, the following function will make this easier.

```@docs
tensor_product_patch
```

It is also possible to obtain the skeleton patch for a given patch type. 
That is, obtaining the patch type of which the boundaries of the given patch type are made. 
This can be done using the following function:

```@docs
get_skeleton_patch
```

## Numbering and Positions

### Mesh numbering
The mesh numbering is defined by `MeshCore`. 

### Patch numbering
The patch numbering for each patch type is shown in their respective docs. 
See [the section on the available patch types](@ref TopologyPatchTypes) for the specifics.

Every patch that is a subtype of [`AbstractTensorProductPatch`](@ref) also supports a position based id next to the generic integer ids.
The position id is expressed as an `NTuple{manifold_dim, Int}`, with each entry being `-1`, `0`, or `+1`.
Each entry encodes the position of the geometric object along each reference axis:
- `-1`: object is located at the minimum of that axis.
- `+1`: object is located at the maximum of that axis.
- `0`: object  extends along that axis (e.g. an edge aligned with it).

For example:

1D elements:
- Vertex 1: position `(-1,)` (at ξ = -1)
- Vertex 2: position `(1,)` (at ξ = +1)
- Edge 1: position `(0,)` (extends along ξ)

2D elements:
- Vertex 1: position `(-1, -1)` (at ξ = -1, η = -1)
- Edge 1: position `(-1, 0)` (at ξ = -1, extends along η)
- Edge 3: position `(0, -1)` (extends along ξ, at η = -1)
- Face 1: position `(0, 0)` (extends along the two axis ξ and η)

3D elements:
- Vertex 1: position `(-1, -1, -1)` (at ξ = -1, η = -1, ζ = -1)
- Edge 4: position `(0, 1, -1)` (extends along ξ, at η = +1, ζ = -1)
- Face 1: position `(-1, 0, 0)` (at ξ = -1, extends along η and ζ)
- Volume 1: position `(0, 0, 0)` (extends along the three axis, ξ, η, and ζ)

You can always convert between these two ids using the following functions.

```@docs
id_to_position
position_to_id
```

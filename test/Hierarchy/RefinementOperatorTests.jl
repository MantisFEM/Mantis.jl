import Mantis

using Test

const Patch1D = Mantis.Mesh.Patch1D
const KnotVector = Mantis.FiniteElementSpaces.KnotVector
const BSplineSpace = Mantis.FiniteElementSpaces.BSplineSpace

# Piece-wise degree of the basis functions on which the tests are performed.
const degrees_to_test = 0:25


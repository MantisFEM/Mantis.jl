module PatchIntervalTests

import Mantis

using Test

Patch = Mantis.Mesh.Patch

n1 = 5
n2 = 8
n3 = 6
n4 = 3
test_brk = collect(LinRange(0.0, 1.0, n1))
test_brk2 = collect(LinRange(-0.25, 2.25, n2))
test_brk3 = collect(LinRange(-1.0, 0.0, n3))
test_brk4 = collect(LinRange(-10.0, -8.0, n4))

test_brk_wrong_order = [0.0, -1.0, 2.0]
test_brk_wrong_order2 = [0.0, 0.0, 1.0]

# The break points are not strictly increasing, which should be checked 
# when creating the Patch and should work also if other breakpoints are 
# correct. Note that the 1D Patch constructor uses a convience 
# constructor, so both the convenience constructor and the general 
# version are used to make sure they work.
@test_throws ArgumentError Patch(test_brk_wrong_order)  # Convenience constructor
@test_throws ArgumentError Patch((test_brk_wrong_order2,))  # General constructor
@test_throws ArgumentError Patch((test_brk, test_brk_wrong_order))
@test_throws ArgumentError Patch((test_brk_wrong_order, test_brk_wrong_order2))
@test_throws ArgumentError Patch((test_brk, test_brk2, test_brk_wrong_order2))

# Patch creation
test_patch_1d = Patch(test_brk)
test_patch = Patch((test_brk, test_brk2, test_brk3, test_brk4))

# Tests for the breakpoints getter
@test Mantis.Mesh.get_breakpoints(test_patch_1d) == (test_brk,)
@test Mantis.Mesh.get_breakpoints(test_patch) == (test_brk, test_brk2, test_brk3, test_brk4)
@test Mantis.Mesh.get_breakpoints(test_patch_1d, 1) == test_brk
@test Mantis.Mesh.get_breakpoints(test_patch, 1) == test_brk
@test Mantis.Mesh.get_breakpoints(test_patch, 2) == test_brk2
@test Mantis.Mesh.get_breakpoints(test_patch, 3) == test_brk3
@test Mantis.Mesh.get_breakpoints(test_patch, 4) == test_brk4

# Test if the size of a Patch is correctly computed. Note that these are 
# redefinitions of Base.size so that one can simply call size() on a Patch.
@test size(test_patch_1d) == (n1-1,)
@test size(test_patch) == (n1-1, n2-1, n3-1, n4-1)
@test size(test_patch_1d, 1) == n1-1
@test size(test_patch, 1) == n1-1
@test size(test_patch, 2) == n2-1
@test size(test_patch, 3) == n3-1
@test size(test_patch, 4) == n4-1

# Test if the element ids are correctly created.
check_ids = NTuple{4, Int}[]
for i in 1:1:(n4 - 1)
    for j in 1:1:(n3 - 1)
        for k in 1:1:(n2 - 1)
            for l in 1:1:(n1 - 1)
                push!(check_ids, (l, k, j, i))
            end
        end
    end
end
for (i, elem_id) in enumerate(Mantis.Mesh.get_element_ids(test_patch))
    @test elem_id == check_ids[i]
end

end

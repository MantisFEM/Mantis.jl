module AbstractTests

using Mantis
using Test

using Mantis.FunctionSpaces: CanonicalBuffer

for T in (Float64, Int, Float32, ComplexF64)
    @test FunctionSpaces.CanonicalValues{T} == Array{T, 3}
end

wrong_type = ["for" "a"; "unit" "test"]
@test_throws MethodError CanonicalBuffer(wrong_type)
wrong_type = Matrix{Float64}(undef, (2, 2))
@test_throws MethodError CanonicalBuffer(wrong_type)

values = fill(1.0, (4, 2, 1))
buff = CanonicalBuffer(values)
@test buff() === values
Caching.clear!(buff)
@test buff() == zeros(4, 2, 1)

new_buff = CanonicalBuffer(ones(4, 2, 1))
copyto!(new_buff, buff)
@test new_buff() == buff()
@test new_buff() == zeros(4, 2, 1)

end

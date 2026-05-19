module AbstractTests

using Mantis
using Test
using Mantis.Geometry: GeometryBuffer

for T in (Float64, Int, Float32, ComplexF64)
    @test Geometry.Values{T} == Matrix{T}
end

wrong_type = ["for" "a"; "unit" "test"]
@test_throws TypeError GeometryBuffer(wrong_type)

values = Geometry.Values{Float64}(undef, 4, 2)
buff = GeometryBuffer(values)
@test Caching.isfilled(buff) == false
Caching.clear!(buff)
@test buff() == zeros(4, 2)

new_buff = GeometryBuffer(ones(4, 2))
copyto!(new_buff, buff)
@test new_buff() == buff()

end

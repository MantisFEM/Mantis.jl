module PlotTests

using Test

@testset "GeometryPlot" begin
    include("GeometryPlotTests.jl")
end
@testset "FormPlot" begin
    include("FormPlotTests.jl")
end

end

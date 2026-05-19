module GeneralHelpersTests

using Mantis
using Test

@testset "export_path" begin
    new_directory = [pwd(), "new", "directory"]
    new_file = "new_file.vtu"
    another_file = "another_file.vtu"
    output_file = Mantis.GeneralHelpers.export_path(new_directory, new_file)
    # Create a new directory
    @test isdir(joinpath(new_directory...))
    @test joinpath(new_directory..., new_file) == output_file
    # Re-use a directory
    output_file = Mantis.GeneralHelpers.export_path(new_directory, another_file)
    @test joinpath(new_directory..., another_file) == output_file
    # Remove created directory
    rm(joinpath(new_directory[1:(end - 1)]...); recursive=true)
end

@testset "num_der_indices" begin
    n = 1
    for i in 0:5
        @test Mantis.GeneralHelpers.num_der_indices(n, i) == 1
    end

    d = 1
    for i in 1:5
        @test Mantis.GeneralHelpers.num_der_indices(i, d) == i
        @test Mantis.GeneralHelpers.num_der_indices(i, 0) == 1
    end

    n = 2
    @test Mantis.GeneralHelpers.num_der_indices(n, 2) == 3
    @test Mantis.GeneralHelpers.num_der_indices(n, 3) == 4
    @test Mantis.GeneralHelpers.num_der_indices(n, 4) == 5

    n = 3
    @test Mantis.GeneralHelpers.num_der_indices(n, 2) == 6
    @test Mantis.GeneralHelpers.num_der_indices(n, 3) == 10
    @test Mantis.GeneralHelpers.num_der_indices(n, 4) == 15
end

end

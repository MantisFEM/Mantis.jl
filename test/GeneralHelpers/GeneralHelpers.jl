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
	rm(joinpath(new_directory[1:end-1]...); recursive=true)
end

end

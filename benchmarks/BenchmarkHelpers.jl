using BenchmarkTools
using CSV
using DataFrames

const rtol = 0.05
const mantis_dir = joinpath(splitpath(pwd())[1:(end - 1)])

"""
    run_benchmark(
        benchmark::BenchmarkTools.Benchmark,
        location::String,
        name::String;
        show=false,
        save=false,
        rtol=rtol,
    )

Runs the benchmarkable object `benchmark`. Optionally, the flags `show` and `save` can be
set to `true` to display the ran benchmark and/or store in a CSV file, respectively. In the
latter case, `location` will be used to determine where the file is stored and `rtol` is the
cutoff determining if a benchmark is deemed worse than before. Note that `location` should
almost always be set automatically with `relpath(dirname(@__FILE__), pwd())` in the file
calling this function. This ensures that the folder structure in `data/` matches the
benchmarks' folder structure.

# Arguments
- `benchmark::BenchmarkTools.Benchmark`: The benchmarkable object.
- `location::String`: The file folder location.
- `name::String`: The name of the benchmark.
- `show=false`: Flag to display benchmark results.
- `save=false`: Flag to store the benchmark results.
- `rtol=0.05`: Tolerance for benchmark performance comparison.

# Returns
- `::Nothing`.
"""
function run_benchmark(
    benchmark::BenchmarkTools.Benchmark,
    location::String,
    name::String;
    show=false,
    save=false,
    rtol=rtol,
)
    if !show && !save
        printstyled(
            "Warning: Both `show` and `save` are set to `false`. " *
            "Nothing will be done.\n";
            color=:red,
        )

        return nothing
    elseif !save
        input = ""
        first = true
        cases = ("y", "n", "c")
        while !(input in cases)
            if !first
                println("Invalid input. Valid options are `y`, `n` or `c`.")
            end
            printstyled(
                "Warning: `save` is set to `false`. No benchmarks will be stored. " *
                "Do you wish to proceed? (y)es, (n)o, (c)hange save to `true`\n";
                color=:yellow,
            )
            input = readline()
            first = false
        end

        if input == "c"
            save = true
        elseif input == "n"
            println("Quitting...")

            return nothing
        end
    end

    hostname, date, commit_hash, commit_date, julia_version = get_metadata()
    columns = [
        :benchmark,
        :commit_hash,
        :commit_date,
        :julia_version,
        :date,
        :med_time,
        :mea_time,
        :std_time,
        :min_time,
        :min_gc,
        :min_memory,
        :min_allocs,
    ]
    # Initialize empty DataFrame with all columns typed as String
    df = DataFrame((col => String[] for col in columns)...)
    dataframe = _run_benchmark!(
        df,
        benchmark,
        name,
        location,
        date,
        commit_hash,
        commit_date,
        julia_version;
        show=show,
        save=save,
    )

    if save
        file_name = "host-$(hostname)-$(name).csv"
        save_results(dataframe, location * "/" * file_name; rtol=rtol)
    end

    return nothing
end

"""
    get_metadata()

Returns useful metadata for CSV storage.

# Returns
- `hostname<:AbstractString`: The hostname of the device used to run the code.
- `date<:AbstractString`: The current date in the format `YYYY-MM-DD`.
- `commit_hash<:AbstractString`: The hash of latest commit in the current branch.
- `julia_version<:AbstractString`: The current julia version.
"""
function get_metadata()
    date = string(readchomp(`date +%Y-%m-%d`))
    long_hash = string(readchomp(`git merge-base remotes/origin/main HEAD`))
    commit_hash = string(readchomp(`git rev-parse --short=7 $long_hash`))
    commit_date = string(
        split(string(readchomp(`git show -s --format="%ci" $commit_hash`)))[1]
    )
    hostname_shasum = split(readchomp(`sh -c "hostname | sha256sum"`))[1]
    julia_version = string(VERSION)
    # Mapping of SHA-256 hashes to hostnames
    hostname_map = Dict(
        "56960c953864e4d3e91ebb242bb87fd03002c2c6e3a62c8da21c719226bd1860" => "DiogoCabanas",
        "2b14f204f7c65a42a64568df8dc0179305ff91886e023977afef7561811ea920" => "JoeyDekker",
        "a04d9fd111db36fac6c5b3923517a6ad605493e46418c6187fee374fa9c37fad" => "DeepeshToshniwal",
        "4d5e2c0292fb7f3c7abc69908372c30adc6547fc8aa52ec4ac0d3ab9f0cc1a51" => "ArturPalha",
        # Add more mappings here as needed
    )

    if haskey(hostname_map, hostname_shasum)
        hostname = hostname_map[hostname_shasum]
    else
        throw(
            ArgumentError(
                "Hostname sha256sum not recognized. " *
                "Run `hostname | sha256sum` in your terminal to get the sha256sum and " *
                "add it to the `hostname_map` dictionary with the desired hostname.",
            ),
        )
    end

    return hostname, date, commit_hash, commit_date, julia_version
end

"""
    _run_benchmark!(
        dataframe,
        group::BenchmarkGroup,
        path::Vector{String},
        hostname::String,
        date::String,
        commit_hash::String,
        julia_version::String;
        show::Bool,
        save::Bool,
    )

Helper method to append a benchmark's results to `dataframe`.

# Arguments
- `dataframe`: The dataframe storing the benchmarks.
- `benchmark::BenchmarkTools.Benchmark`: The benchmark being ran.
- `name::String`: The benchmark's name.
- `location::String`: The benchmark's location.
- `date::String`: The current date in the format `YYYY-MM-DD`.
- `commit_hash::String`: The hash of latest commit in the current branch.
- `julia_version::String`: The current julia version.
- `show=false`: Flag to display benchmark results.
- `save=false`: Flag to store the benchmark results.

# Returns
- `dataframe`: Updated dataframe at current nest level.
"""
function _run_benchmark!(
    dataframe,
    benchmark::BenchmarkTools.Benchmark,
    name::String,
    location::String,
    date::String,
    commit_hash::String,
    commit_date::String,
    julia_version::String;
    show::Bool,
    save::Bool,
)
    full_name = joinpath(location, name)
    println("Benchmark: $(full_name)")
    #= The tuning should be uncommented if the benchmarking parameters are not set.
    print(indent * "Tuning...")
    tune!(val)
    print("Done! ") =#
    print("Running...")
    result = run(benchmark)
    print("Done! ")
    if show
        println()
        display(result)
    end

    if save
        push!(
            dataframe,
            (
                benchmark=full_name,
                commit_hash=commit_hash,
                commit_date=commit_date,
                julia_version=julia_version,
                date=date,
                med_time=BenchmarkTools.prettytime(median(result).time),
                mea_time=BenchmarkTools.prettytime(mean(result).time),
                std_time=BenchmarkTools.prettytime(std(result).time),
                min_time=BenchmarkTools.prettytime(minimum(result).time),
                min_gc=BenchmarkTools.prettytime(minimum(result).gctime),
                min_memory=BenchmarkTools.prettymemory(minimum(result).memory),
                min_allocs=string(minimum(result).allocs),
            ),
        )
    end

    return dataframe
end

"""
    save_results(new_dataframe::DataFrame, file_name::String; rtol=0.05)

Stores the benchmark results in `new_dataframe` in the CSV file `file_name`. The tolerance
`rtol` is the cutoff determining if a benchmark is worse than before.

# Arguments
- `new_dataframe::DataFrame`: The dataframe containing the benchmark results.
- `file_name::String`: String containing the full file path.
- `rtol=0.05`: The cutoff tolerance for worse benchmarks.

# Returns
- `::Nothing`.
"""
function save_results(new_dataframe::DataFrame, file_name::String; rtol=0.05)
    print("Saving to CSV...")
    if !endswith(file_name, ".csv")
        throw(ArgumentError("Invalid file name. Must end with `.csv`."))
    end

    data_prefix = "data"
    if isfile(joinpath(data_prefix, file_name))
        types_dict = Dict(col => String for col in names(new_dataframe))
        data_filename = joinpath(data_prefix, file_name)
        old_dataframe = CSV.read(data_filename, DataFrame; types=types_dict)
        duplicate_rows, new_rows = get_duplicate_and_new_rows(old_dataframe, new_dataframe)
        num_duplicate = length(duplicate_rows)
        if num_duplicate != 0
            print("Duplicate found. ")
        end

        for (old_id, new_row) in duplicate_rows
            old_row = old_dataframe[old_id, :]
            println("Comparing...")
            str_old_time = old_row.min_time
            str_new_time = new_row.min_time
            ratio = compare_times(str_old_time, str_new_time)
            if ratio < 1.0
                printstyled(
                    "Minimum time improved! Old time=" *
                    str_old_time *
                    ", new time=" *
                    str_new_time *
                    ", ratio=$(ratio).\n";
                    color=:green,
                )
                for col in names(new_row)
                    old_dataframe[old_id, col] = new_row[col]
                end
            elseif ratio <= 1.0 + rtol
                printstyled(
                    "Minimum time increased within $(rtol) tolerance. Old time=" *
                    str_old_time *
                    ", new time=" *
                    str_new_time *
                    ", ratio=$(ratio).\n";
                    color=:yellow,
                )
            else
                printstyled(
                    "Minimum time increased outside $(rtol) tolerance! Old time=" *
                    str_old_time *
                    ", new time=" *
                    str_new_time *
                    ", ratio=$(ratio).\n";
                    color=:red,
                )
            end
        end

        if !isempty(new_rows)
            append!(old_dataframe, new_rows)
        end

        CSV.write(data_filename, old_dataframe)
    else
        mkpath(joinpath(data_prefix, dirname(file_name)))
        CSV.write(joinpath(data_prefix, file_name), new_dataframe)
    end

    println("Done!")

    return nothing
end

"""
    get_duplicate_and_new_rows(old_dataframe, new_dataframe)

Returns the new and duplicated rows in `new_dataframe` when compared to `old_dataframe`.

# Arguments
- `old_dataframe`: The old dataframe.
- `new_dataframe`: The new dataframe.

# Returns
- `duplicate_rows::Vector{Tuple{Int, DataFrameRow{DataFrame, DataFrames.Index}}}`: The row
    id of a duplicated rows in `old_dataframe` and the equivalent row in `new_dataframe`.
- `new_rows::DataFrameRow{DataFrame, DataFrames.Index}`: The new rows.
"""
function get_duplicate_and_new_rows(old_dataframe, new_dataframe)
    duplicate_rows = Tuple{Int, DataFrameRow{DataFrame, DataFrames.Index}}[]
    new_rows = DataFrameRow{DataFrame, DataFrames.Index}[]

    for new_row in eachrow(new_dataframe)
        found_duplicate = false
        for (id, old_row) in pairs(IndexLinear(), eachrow(old_dataframe))
            if is_duplicate(old_row, new_row)
                push!(duplicate_rows, (id, new_row))
                found_duplicate = true
                break
            end
        end

        if !found_duplicate
            push!(new_rows, new_row)
        end
    end

    return duplicate_rows, new_rows
end

"""
    is_duplicate(old_row, new_row)

Determines if `old_row` and `new_row` refer to the same benchmark by comparing the name and
commit hash.

# Arguments
- `old_row`: The old row.
- `new_row`: The new row.

# Returns
- `::Bolean`: Whether the new row is a duplicate or not.
"""
function is_duplicate(old_row, new_row)
    if old_row.benchmark != new_row.benchmark
        return false
    end

    if old_row.commit_hash != new_row.commit_hash
        return false
    end

    return true
end

"""
    compare_times(str_old_time::AbstractString, str_new_time::AbstractString)

Compares the times found in two strings using BenchmarkTools' pretty-print style.

# Arguments
- `str_old_time::AbstractString`: The old benchmark time.
- `str_new_time::AbstractString`: The new benchmark time.

# Returns
- `ratio::Float64`: The ratio of the new time over the old time.
"""
function compare_times(str_old_time::AbstractString, str_new_time::AbstractString)
    # Get the numbers as floats in the same unit
    old_time = get_number_in_string(str_old_time; new_unit="s")
    new_time = get_number_in_string(str_new_time; new_unit="s")
    ratio = new_time / old_time

    return ratio
end

"""
    get_number_in_string(str::AbstractString; new_unit::String)

Returns the number found in `str` as a `Float64`. The number is given in the same unit as
found in the string, unless otherwise specified by `new_unit`. This method assumes the
string is given in the pretty-print, number and unit, format used in `BenchmarkTools`, for
example, `12.345 ms`.

# Arguments
- `str::AbstractString`: The pretty-print string.
- `new_unit::String`: The new unit in which the time is returned.

# Returns
- `::Float64`: The number found in `str`.
"""
function get_number_in_string(str::AbstractString; new_unit::Union{Nothing, String}=nothing)
    # Conversion factors from different units to second
    factors = Dict(
        "ns" => 1e-9,  # Time
        "μs" => 1e-6,
        "ms" => 1e-3,
        "s" => 1,
        "bytes" => 1,  # Memory
        "KiB" => 1024,
        "MiB" => 1024^2,
        "GiB" => 1024^3,
    )
    str_number, str_unit = split(str) # Assumes "number unit" format of the string
    number = parse(Float64, str_number)
    if !isnothing(new_unit)
        number_in_seconds = number * factors[str_unit]
        number = number_in_seconds / factors[new_unit]
    end

    return number
end

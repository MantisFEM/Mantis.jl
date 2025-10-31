using GLMakie
using CSV
using DataFrames
using Dates
using InlineStrings

include("BenchmarkHelpers.jl")


const colours = Dict(
    "JoeyDekker" => :orange,
    "DiogoCabanas" => :blue,
    "ArturPalha" => :green,
    "DeepeshToshniwal" => :red,
)

const markers = Dict(
    "JoeyDekker" => :circle,
    "DiogoCabanas" => :cross,
    "ArturPalha" => :diamond,
    "DeepeshToshniwal" => :star6,
)

const units = Dict(
    :min_time => "ms",
    :med_time => "ms",
    :std_time => "ms",
    :mea_time => "ms",
    :min_memory => "MiB",
    :min_allocs => "#"
)

data = Dict{Tuple{String, String}, DataFrame}()


data_dir = joinpath(pwd(), "data")
benchmark_series_vec = String[]
for (dirpath, dirnames, filenames) in walkdir(data_dir)
    if length(filenames) > 0
        push!(benchmark_series_vec, dirpath)
    end
end
unique!(sort!(benchmark_series_vec))

function get_benchmarks(benchmark_series)
    benchmarks = String[]
    benchmarknames = String[]
    hostnames = String[]
    for (dirpath, dirnames, filenames) in walkdir(benchmark_series)
        for filename in filenames
            if endswith(filename, ".csv")
                # Get the benchmark name without the extension and host name
                name_parts = split(filename, "-")
                hostname = name_parts[2]
                benchmarkname = join(name_parts[3:end], "-")[1:end-4]
                push!(benchmarks, joinpath(dirpath, filename))
                push!(benchmarknames, benchmarkname)
                push!(hostnames, hostname)
            end
        end
    end

    unique!(sort!(benchmarks))
    unique!(sort!(benchmarknames))
    unique!(sort!(hostnames))

    return benchmarks, benchmarknames, hostnames
end
notify(menu_unit.selection)

function update_data!(data, name, dirname, hostnames)
    for hostname in hostnames
        filename = "host-$hostname-$name.csv"
        println("Updating plotting data using file: $filename")
        df = DataFrame(CSV.File(joinpath(dirname, filename)))
        data[(hostname, name)] = df
    end
    return data
end


# Create a root observable with the benchmark series name. Whenever this changes, everything
# else updates automatically. Set the first benchmark as default.
benchmark_series_ob = Observable(benchmark_series_vec[1])
benchmark_names_ob = Observable(get_benchmarks(to_value(benchmark_series_ob))[2])
benchmark_name_ob = Observable(to_value(benchmark_names_ob)[1])

update_data!(
    data,
    to_value(benchmark_name_ob),
    to_value(benchmark_series_ob),
    get_benchmarks(to_value(benchmark_series_ob))[3]
)
benchmark_data_ob = Observable(data)



fig = Figure()

# Menus:
# Benchmark series menu
menu_benchmark_series = Menu(
    fig, options = benchmark_series_vec, default = to_value(benchmark_series_ob)
)
# Benchmark menu
menu_benchmark_names = Menu(
    fig, options = benchmark_names_ob, default = to_value(benchmark_name_ob)
)
# X data menu
menu_x_data = Menu(fig, options = ["commit_date", "date"], default = "commit_date")
# Y data menu
menu_y_data = Menu(
    fig,
    options = ["min_time", "med_time", "std_time", "mea_time", "min_memory", "min_allocs"],
    default = "min_time"
)

col_x = Observable(:commit_date)
col_y = Observable(:min_time)

# Create a list of menus on the top of the figure.
fig[1, 1] = vgrid!(
    Label(fig, "Benchmark Series", width = nothing),
    menu_benchmark_series,
    Label(fig, "Benchmark", width = nothing),
    menu_benchmark_names;
    tellheight = false, width = 1000
)

fig[1, 2] = vgrid!(
    Label(fig, "X-data", width = nothing),
    menu_x_data,
    Label(fig, "Y-data", width = nothing),
    menu_y_data;
    tellheight = false, width = nothing
)



ax = Axis(
    fig[2, 1:2],
    xlabel = "Commit date [yyyy-mm-dd]",
    ylabel = "Time [ms]",
    xticklabelrotation=45.0
)

all_dates = lift(benchmark_data_ob, benchmark_name_ob, col_x) do data, name, col
    all_dates = vcat(
        data[("JoeyDekker", name)][!, col],
        data[("DiogoCabanas", name)][!, col],
        data[("ArturPalha", name)][!, col],
        data[("DeepeshToshniwal", name)][!, col]
    )
    all_dates_strings = Dates.format.(all_dates, "yyyy-mm-dd")
    unique_dates = unique(sort(all_dates))
    diffs = Dates.value.(unique_dates .- first(unique_dates))
    ax.xticks = (diffs, Dates.format.(unique_dates, "yyyy-mm-dd"))
    xlims!(ax, minimum(diffs)-1, maximum(diffs)+1)
    return unique_dates
end

x_data_joey = lift(all_dates) do dates
    data = to_value(benchmark_data_ob)
    name = to_value(benchmark_name_ob)
    col = to_value(col_x)
    all_dates_joey = data[("JoeyDekker", name)][!, col]
    all_dates_diff = dates .- first(dates)
    idxs = findall(x -> x in all_dates_joey, dates)
    return Dates.value.(all_dates_diff[idxs])
end
x_data_diogo = lift(all_dates) do dates
    data = to_value(benchmark_data_ob)
    name = to_value(benchmark_name_ob)
    col = to_value(col_x)
    all_dates_diogo = data[("DiogoCabanas", name)][!, col]
    all_dates_diff = dates .- first(dates)
    idxs = findall(x -> x in all_dates_diogo, dates)
    return Dates.value.(all_dates_diff[idxs])
end
x_data_artur = lift(all_dates) do dates
    data = to_value(benchmark_data_ob)
    name = to_value(benchmark_name_ob)
    col = to_value(col_x)
    all_dates_artur = data[("ArturPalha", name)][!, col]
    all_dates_diff = dates .- first(dates)
    idxs = findall(x -> x in all_dates_artur, dates)
    return Dates.value.(all_dates_diff[idxs])
end
x_data_deepesh = lift(all_dates) do dates
    data = to_value(benchmark_data_ob)
    name = to_value(benchmark_name_ob)
    col = to_value(col_x)
    all_dates_deepesh = data[("DeepeshToshniwal", name)][!, col]
    all_dates_diff = dates .- first(dates)
    idxs = findall(x -> x in all_dates_deepesh, dates)
    return Dates.value.(all_dates_diff[idxs])
end



y_data_joey = lift(benchmark_data_ob, benchmark_name_ob, col_y) do data, name, col
    plot_data = data[("JoeyDekker", name)][:, col]
    if eltype(plot_data) <: AbstractString || eltype(plot_data) <: InlineString
        return get_number_in_string.(data[("JoeyDekker", name)][:,col], new_unit=units[col])
    else
        return plot_data
    end
end
y_data_diogo = lift(benchmark_data_ob, benchmark_name_ob, col_y) do data, name, col
    plot_data = data[("DiogoCabanas", name)][:, col]
    if eltype(plot_data) <: AbstractString || eltype(plot_data) <: InlineString
        return get_number_in_string.(data[("DiogoCabanas", name)][:,col], new_unit=units[col])
    else
        return plot_data
    end
end
y_data_artur = lift(benchmark_data_ob, benchmark_name_ob, col_y) do data, name, col
    plot_data = data[("ArturPalha", name)][:, col]
    if eltype(plot_data) <: AbstractString || eltype(plot_data) <: InlineString
        return get_number_in_string.(data[("ArturPalha", name)][:,col], new_unit=units[col])
    else
        return plot_data
    end
end
y_data_deepesh = lift(benchmark_data_ob, benchmark_name_ob, col_y) do data, name, col
    plot_data = data[("DeepeshToshniwal", name)][:, col]
    if eltype(plot_data) <: AbstractString || eltype(plot_data) <: InlineString
        return get_number_in_string.(data[("DeepeshToshniwal", name)][:,col], new_unit=units[col])
    else
        return plot_data
    end
end

scatterlines!(ax, x_data_joey, y_data_joey;
    label="JoeyDekker", color=colours["JoeyDekker"], marker=markers["JoeyDekker"]
)
scatterlines!(ax, x_data_diogo, y_data_diogo;
    label="DiogoCabanas", color=colours["DiogoCabanas"], marker=markers["DiogoCabanas"]
)
scatterlines!(ax, x_data_artur, y_data_artur;
    label="ArturPalha", color=colours["ArturPalha"], marker=markers["ArturPalha"]
)
scatterlines!(ax, x_data_deepesh, y_data_deepesh;
    label="DeepeshToshniwal", color=colours["DeepeshToshniwal"], marker=markers["DeepeshToshniwal"]
)

axislegend("Host names", position = :rt)


# Define what happens when a menu selection changes.
on(menu_benchmark_series.selection) do s
    println("Selected benchmark series: $s")
    benchmark_series_ob[] = s
    benchmark_names_ob[] = get_benchmarks(s)[2]
end

on(menu_benchmark_names.selection) do s
    if !isnothing(s)
        println("Selected benchmark name: $s")
        benchmark_stringdata = get_benchmarks(benchmark_series_ob[])
        update_data!(data, s, to_value(benchmark_series_ob), benchmark_stringdata[3])
        benchmark_name_ob[] = s
        ax.title = s
        autolimits!(ax)
    end
end

on(menu_x_data.selection) do s
    println("Selected x data: $s")
    col_x[] = Symbol(s)
    if Symbol(s) == :commit_date
        ax.xlabel = "Commit date [yyyy-mm-dd]"
    else
        ax.xlabel = "Date [yyyy-mm-dd]"
    end
end

on(menu_y_data.selection) do s
    println("Selected y data: $s")
    col_y[] = Symbol(s)
    if Symbol(s) in [:min_time, :median_time, :std_time, :mean_time]
        ax.ylabel = "Time [ms]"
    elseif Symbol(s) == :min_memory
        ax.ylabel = "Memory [MiB]"
    elseif Symbol(s) == :min_allocs
        ax.ylabel = "Allocations [#]"
    end
    autolimits!(ax)
end

display(fig)

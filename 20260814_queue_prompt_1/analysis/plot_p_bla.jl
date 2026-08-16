using GLMakie
using CSV
using DataFrames

BASE_DIR = dirname(@__DIR__)
INTERP_HOME = dirname(BASE_DIR)
DATA_DIR = joinpath(INTERP_HOME, "data_library", "generated")

# Edit these to compare different runs.
DATASETS = [
    ("95-batch run (Aug 15, 09:20)", "prompt_file_2026-08-15T09:20:22.381.csv"),
    ("965-batch run (Aug 15, 20:54)", "prompt_file_2026-08-15T20:54:10.219.csv"),
]

fig = Figure(size = (900, 500))
ax = Axis(
    fig[1, 1],
    xlabel = "p(Black) [BISG]",
    ylabel = "density",
    title = "Distribution of p_bla by dataset",
)

colors = [:dodgerblue, :orangered]

for ((label, filename), color) in zip(DATASETS, colors)
    df = CSV.read(joinpath(DATA_DIR, filename), DataFrame)
    hist!(
        ax, df.p_bla;
        bins = range(0, 1, length = 51),
        normalization = :pdf,
        color = (color, 0.45),
        strokecolor = color,
        strokewidth = 1,
        label = "$label (n=$(nrow(df)))",
    )
end

axislegend(ax, position = :rt)

output_path = joinpath(@__DIR__, "p_bla_distribution.png")
save(output_path, fig)
println("wrote ", output_path)

display(fig)

using CSV
using DataFrames
using StatsBase

using GLMakie

manifest_file = joinpath(
    "challenge_tables_2026-09-11T08:30:23.040625",
    "manifest.csv",
)
manifest_path = joinpath(dirname(@__DIR__), "input_data", manifest_file)
manifest = CSV.read(manifest_path, DataFrame)

outcome_file = "outcomes_2026-09-12T12_58_56.274506.csv"
outcome_path = joinpath(dirname(@__DIR__), "results", outcome_file)
frame = CSV.read(outcome_path, DataFrame)

conts = innerjoin(frame, manifest, on = :experiment_id => :fileid) |>
# m -> filter(:batch => ==(2), m) |>
m -> select(m, [:batch, :race, :position, :continuation])


function parse_outcome(cont)
    numbers = first(split(cont, "\n", limit=2))
    return something.(tryparse.(Int, split(strip(numbers), ",")), missing)
end

conts.parsed_cont = parse_outcome.(conts.continuation)

conts.rank = [something(findfirst(isequal(p), c), missing) for (p, c) in zip(conts.position, conts.parsed_cont)]


batch_mean = combine(groupby(
    dropmissing(conts),
    [:batch],
), :rank => mean => :batch_avg)

race_batch_mean = combine(groupby(
    dropmissing(conts),
    [:batch, :race],
), :rank => mean => :race_batch_avg)

withmeans = innerjoin(race_batch_mean, batch_mean, on = :batch)

withmeans = transform(
    withmeans,
    [:batch_avg, :race_batch_avg] => ((a, b) -> a .- b) => :difference,
) 

transform(
    withmeans,
    [:batch_avg, :race_batch_avg] => ((a, b) -> a .- b) => :difference,
) |> f -> combine(groupby(f, :race), :difference => (x -> quantile(x, [0.05, 0.1, 0.9, 0.95])))



races = ["bla", "whi", "asi", "his", "oth"]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "race_batch_avg − $(check)")
for (i, r) in enumerate(races)

    match_mean = race_batch_mean[race_batch_mean.race .== r, :]
    nonmatch_mean = race_batch_mean[race_batch_mean.race .!= r, :]

    points = innerjoin(match_mean, nonmatch_mean, on = :batch, makeunique=true) |>
    frame -> frame.race_batch_avg_1 .- frame.race_batch_avg

    density!(ax, points;
             label = r, color = (:transparent, 0),
             strokecolor = Makie.wong_colors()[i], strokewidth = 2)

end

axislegend(ax)
fig

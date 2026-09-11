using CSV
using DataFrames
using StatsBase


frame = CSV.read(joinpath(@__DIR__, "..", "data", "token_counts.csv"), DataFrame)

# Group by + summarize
avgs = combine(groupby(frame, :top), :n_tokens => mean => :avg)
stds = combine(groupby(frame, :top), :n_tokens => std => :std)

groups = leftjoin(avgs, stds, on = :top)

display(groups)


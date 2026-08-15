using DataFrames
using CSV
using Dates

include("../queuemaker/analysis/queries.jl")

BATCHES = 100

names = ohio_voter_names(BATCHES)
loan_details = hmda_sample(BATCHES)
frame = hcat(names, loan_details)
tuples = divrem.((1:nrow(frame)) .- 1, 10)
batch, id = first.(tuples), last.(tuples)
frame.batch = batch
frame.id = id
frame.global_id = 1:nrow(frame)

CSV.write("../data_library/generated/prompt_file_$(now()).csv", frame)

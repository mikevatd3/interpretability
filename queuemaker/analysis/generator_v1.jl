using DataFrames
using CSV
using Dates

include("queries.jl")

BATCHES = 10

names = ohio_voter_names()
loan_details = hmda_sample()
frame = hcat(names, loan_details)
tuples = divrem.((1:nrow(frame)) .- 1, 10)
batch, id = first.(tuples), last.(tuples)
frame.batch = batch
frame.id = id
frame.global_id = 1:nrow(frame)

CSV.write("../../data_library/generated/prompt_file_$(today()).csv", frame)

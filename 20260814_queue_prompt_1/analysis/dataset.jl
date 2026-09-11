using JSON3
using DataFrames
using CSV

BASE_DIR = dirname(@__DIR__)
RESULT_FILE = "results_2026-08-15T21:07:26.144358.json"

result_path = joinpath(
    BASE_DIR,
    "results",
    RESULT_FILE,
)
run_results = JSON3.read(read(result_path, String))

prompt_path = joinpath(
    dirname(BASE_DIR),
    "data_library",
    "generated",
    run_results[:source][:prompt_file]
)

prompts = CSV.read(prompt_path, DataFrame)

prompts.borrower_income = max.(prompts.borrower_income, 0.0)


# Adding rank column ------------------------------------------------------------

# batch[:order][k] is the (0-indexed) within-batch row id of the item ranked
# k-th; invert that into a 1-indexed rank per row id.
rank_lookup = Dict(
    b[:batch] => begin
        r = similar(b[:order]) # similar creates a mat or vec with same shape and type
        for (k, id) in enumerate(b[:order])
            r[id + 1] = k
        end
        r
    end
    for b in run_results[:batches]
)

prompts.rank = Vector{Union{Missing, Int}}(missing, nrow(prompts))
for g in groupby(prompts, :batch)
    batch_id = first(g.batch)
    if haskey(rank_lookup, batch_id)
        g.rank .= rank_lookup[batch_id]
    end
end

prompts.rank = Int.(prompts.rank)


# Scaling features -------------------------------------------------------------

prompts.log_income = log1p.(max.(prompts.borrower_income, 0))
prompts.log_loan_amount = log1p.(prompts.loan_amount)
prompts.log_property_value = log1p.(prompts.property_value)

zscore(x) = (x .- mean(x)) ./ std(x)

prompts.income_score = zscore(prompts.log_income)
prompts.loan_amount_score = zscore(prompts.log_loan_amount)
prompts.property_value_score = zscore(prompts.log_property_value)

prompts.p_bla = clamp.(prompts.p_bla, 1e-6, 1 - 1e-6)
prompts.logit_bla = log.(prompts.p_bla ./ (1 .- prompts.p_bla))

prompts.p_whi = clamp.(prompts.p_whi, 1e-6, 1 - 1e-6)
prompts.logit_whi = log.(prompts.p_whi ./ (1 .- prompts.p_whi))




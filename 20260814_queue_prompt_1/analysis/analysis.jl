using Turing
using LogExpFunctions
using JSON3
using Distributions
using DataFrames
using CSV


run_results = JSON3.read(read("results/results_2026-08-14T20:15:04.448201.json", String))
prompt_path = "../data_library/generated/" * run_results[:source][:prompt_file]

prompts = CSV.read(prompt_path, DataFrame)

# Scaling features -------------------------------------------------------------

prompts.log_income = log1p.(prompts.borrower_income)
prompts.log_loan_amount = log1p.(prompts.loan_amount)
prompts.log_property_value = log1p.(prompts.property_value)

zscore(x) = (x .- mean(x)) ./ std(x)

prompts.income_score = zscore(prompts.log_income)
prompts.loan_amount_score = zscore(prompts.log_loan_amount)
prompts.property_value_score = zscore(prompts.log_property_value)

# Running the model ------------------------------------------------------------

@model function pl_regression(run_results, prompts)
    feature_cols = [:income_score, :loan_amount_score, :property_value_score]
    p = length(feature_cols)
    β ~ MvNormal(zeros(p), 2.0 * I)

    for batch in run_results[:batches]
        input_frame = filter(:batch => b -> b == batch[:batch], prompts)
        σ = batch[:order] .+ 1 # ids are 0-indexed
        Xt = input_frame[:, feature_cols] |> Matrix

        η = Xt * β
        for k in 1:(length(σ) - 1)
            rest = σ[k:end]
            Turing.@addlogprob! η[σ[k]] - logsumexp(η[rest])
        end
    end
end

chain = sample(pl_regression(run_results, prompts), NUTS(), 1000)

betas = chain[:β].data |> vec
betas = hcat(betas...)'


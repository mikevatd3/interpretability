using Turing
using LogExpFunctions
using JSON3
using Distributions
using DataFrames
using CSV
using StatsBase


BASE_DIR = dirname(@__DIR__)
INTERP_HOME = dirname(BASE_DIR)
RESULT_FILE = "results_2026-08-15T21:07:26.144358.json"

result_path = joinpath(BASE_DIR, "results", RESULT_FILE)
run_results = JSON3.read(read(result_path, String))

prompt_path = joinpath(
    INTERP_HOME,
    "data_library",
    "generated",
    run_results[:source][:prompt_file]
)

prompts = CSV.read(prompt_path, DataFrame)

prompts.borrower_income = max.(prompts.borrower_income, 0.0)


# Scaling features -------------------------------------------------------------

prompts.log_income = log1p.(prompts.borrower_income)
prompts.log_loan_amount = log1p.(prompts.loan_amount)
prompts.log_property_value = log1p.(prompts.property_value)

zscore(x) = (x .- mean(x)) ./ std(x)

prompts.income_score = zscore(prompts.log_income)
prompts.loan_amount_score = zscore(prompts.log_loan_amount)
prompts.property_value_score = zscore(prompts.log_property_value)

prompts.p_bla = clamp.(prompts.p_bla, 1e-6, 1 - 1e-6)
prompts.logit_bla = log1p.(prompts.p_bla ./ (1 .- prompts.p_bla))

# Running the model ------------------------------------------------------------

# Group once, up front, into a batch -> feature-matrix lookup so the model
# doesn't re-scan the whole `prompts` frame for every batch on every
# log-density evaluation (that O(batches * rows) filter() choked at ~1000
# batches).
feature_cols = [
    :income_score,
    :loan_amount_score,
    :property_value_score,
    :logit_bla,
]

batch_features = Dict(
    first(g.batch) => Matrix(g[:, feature_cols]) for g in groupby(prompts, :batch)
)

@model function pl_regression(run_results, batch_features, p)
    β ~ MvNormal(zeros(p), 2.0 * I)

    for batch in run_results[:batches]
        σ = batch[:order] .+ 1 # ids are 0-indexed
        Xt = batch_features[batch[:batch]]

        η = Xt * β
        for k in 1:(length(σ) - 1)
            rest = σ[k:end]
            Turing.@addlogprob! η[σ[k]] - logsumexp(η[rest])
        end
    end
end

chain = sample(pl_regression(run_results, batch_features, length(feature_cols)), NUTS(), 1000)

betas = chain[:β].data |> vec
betas = hcat(betas...)'

println("\n--- coefficient summary ---")
for (i, name) in enumerate(["income_score", "loan_amount_score", "property_value_score", "logit_bla"])
    col = betas[:, i]
    println("$name: mean=$(round(mean(col), digits=3))  std=$(round(std(col), digits=3))  95%CI=($(round(quantile(col, 0.025), digits=3)), $(round(quantile(col, 0.975), digits=3)))")
end


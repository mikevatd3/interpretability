using LibPQ
using DataFrames
using GLMakie
using Turing
using Distributions
using LinearAlgebra
using Random
using StatsBase

include("/home/michael/3_labratory/makie_themes/everforest_makie.jl")
set_theme!(theme_everforest_dark())

conn = LibPQ.Connection("host=localhost dbname=hmda user=michael")
result = execute(conn, """
SELECT score_orig_r AS credit_score,
       loan_amount_cat AS loan_amount,
       x83 AS income,
       dti AS debt_to_income
FROM nsmo_puf
WHERE open_year = 2013
AND score_orig_r IS NOT NULL
AND x83 IS NOT NULL
AND loan_amount_cat IS NOT NULL;
""")
frame = DataFrame(result)
frame = dropmissing(frame)
close(conn)
heatmap(counts(frame.income, frame.loan_amount))

income_cutpoints = [
    0,
    35_000,
    50_000,
    75_000,
    100_000,
    175_000,
]

log_income_cutpoints = log.(cutpoints)

loan_amount_cutpoints = [
    50_000,
    100_000,
    150_000,
    200_000,
    250_000,
    300_000,
    350_000,
    400_000,
]

function make_midpointer(cutpoints)
    function bin_midpoints(i)
    end
end



@model function ordinal_known_cutpoints(X, y, cutpoints)
    n, p = size(X)
    β ~ MvNormal(zeros(p), 4.0 * I)
    σ ~ truncated(Normal(0, 2); lower=0)

    η = X * β
    K = length(cutpoints) + 1  # number of ordinal categories

    for i in 1:n
        lo = y[i] == 1 ? -Inf : cutpoints[y[i]-1]
        hi = y[i] == K ? Inf : cutpoints[y[i]]
        p_i = cdf(Normal(η[i], σ), hi) - cdf(Normal(η[i], σ), lo)
        Turing.@addlogprob! log(p_i)
    end
end


model = ordinal_known_cutpoints()

chain = sample(model, NUTS(), 1000)


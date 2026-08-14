using LibPQ
using DataFrames
using Turing
using Distributions
using GLMakie
using StatsBase

include("/home/michael/3_labratory/makie_themes/hokusai_makie.jl")
set_theme!(theme_hokusai())

include("queries.jl")

hmda = hmda_sample()
log_incomes = log.(hmda.income)

@model function hmda_income(log_incomes)
    σ ~ truncated(Normal(0, 2); lower=0)
    μ ~ Normal(15, 4)
    log_incomes .~ Normal(μ, σ)
end

chain = sample(hmda_income(log_incomes), NUTS(), 1000)


μs = vec(chain[:μ])
σs = vec(chain[:σ])

n_draws = 100  # subsample from chain, don't need all iterations
idx = rand(1:length(μs), n_draws)

ppc = [rand(Normal(μs[i], σs[i]), length(log_incomes)) for i in idx]

fig = Figure()
ax = Axis(fig[1,1])
for sim in ppc
    hist!(ax, sim, color=(:gray, 0.1), bins=50, normalization=:pdf)
end
hist!(ax, log_incomes, color=(:blue, 0.5), bins=50, normalization=:pdf)
fig

function qq_dist(x, dist)
    n = length(x)
    sorted_x = sort(x)
    p = ((1:n) .- 0.5) ./ n
    theoretical = quantile.(dist, p)
    return theoretical, sorted_x
end

dist = Normal(mean(chain[:μ]), mean(chain[:σ]))
theoretical, sorted_x = qq_dist(log_incomes, dist)

fig = Figure()
ax = Axis(fig[1,1], xlabel="Theoretical quantiles", ylabel="Sample quantiles")
scatter!(ax, theoretical, sorted_x)
ablines!(ax, 0, 1, color=:red)  # reference line y=x
fig


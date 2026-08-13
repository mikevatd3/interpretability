using Random
using LinearAlgebra
using Distributions
using StatsBase
using GLMakie

include("/home/michael/3_labratory/makie_themes/everforest_makie.jl")



"""
Looking at the regression explanation in Xin et al. (2026), How Proxy Race 
Distorts Regression-Based Fairness Audits.
"""

# p(r|s,g) = p(r|s)p(g|r) / ∑ p(r'|s)p(g|r')

# p(r|s) surname-based probability from the census surname tabulations 
#        typically from the census: https://www.census.gov/topics/population/genealogy/data/2020_names.html
# p(g|r) geography-based probability the proportion of race r in location g 
#        within the entire race population of r I assume using census data
# The ∑ and r' is just summing over all the rs, so you have everything you need 
# from these two sources

# The addition of first names (census also provides this)
#
# Simple error we're looking at: "errors move observations between categories"

# y = Xβ + ϵ -- This is the regression that we'd like to do with a clean X (the categorical matrix)

CATEGORIES = 5
ROWS = 1000

X = zeros(Int, ROWS, CATEGORIES)
X_hat = zeros(Int, ROWS, CATEGORIES)
cats = rand(1:CATEGORIES, ROWS)

rows = []
for i in 1:CATEGORIES
    row = rand(Dirichlet(CATEGORIES, 0.4))
    _, shiftby = findmax(row)
    row = circshift(row, -(shiftby)+i)
    push!(rows, row)
end

confusion_matrix = hcat(rows...)'

for (i, cat) in zip(1:ROWS, cats)
    X[i, cat] = 1
    cat_hat = sample(1:CATEGORIES, weights(confusion_matrix[cat,:]))
    X_hat[i, cat_hat] = 1
end

(((X .!= X_hat) |> eachrow .|> sum .|> x -> x > 0) |> sum) / ROWS

β = rand(Float32, (CATEGORIES, 1)) * 20;
ϵ = rand(Float32, (ROWS, 1)) * 4;

y = X * β .+ ϵ;

β
β_hat = inv(X'*X) * X'*y
β_hat_hat = inv(X_hat'*X_hat) * X_hat'*y

hist(y[:, 1])


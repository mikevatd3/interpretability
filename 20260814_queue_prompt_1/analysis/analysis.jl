using Turing
using LogExpFunctions
using JSON3
using Distributions
using DataFrames
using CSV
using StatsBase

using StatsModels

using GLMakie

include("plackettluce.jl")
include("connectivity.jl")
include("visualize.jl")
include("model.jl")

feature_cols = [
    :income_score,
    :loan_amount_score,
    :property_value_score,
    :logit_whi,
]

ranked_prompts = sort(dropmissing(prompts, :rank), [:batch, :rank])
batch_groups = groupby(ranked_prompts, :batch)

Xts = [Matrix(g[:, feature_cols]) for g in batch_groups]
rankings = [collect(1:nrow(g)) for g in batch_groups]

p = length(feature_cols)
chain = sample(
    pl_regression(rankings, Xts, p),
    NUTS(),
    1000;
    initial_params=InitFromParams((β=zeros(p),)),
)

cities = unique(prompts.property_city)
prompts.city_id = Int.(indexin(prompts.property_city, cities))

city_rankings = combine(groupby(
    sort(prompts, [:batch, :rank]),
    :batch,
), :city_id => (x -> [x]) => :city_ids)

check_connectivity(city_rankings.city_ids, length(cities))

ranked_prompts = sort(dropmissing(prompts, :rank), [:batch, :rank])
batch_groups = groupby(ranked_prompts, :batch)

Xts = [Matrix(g[:, feature_cols]) for g in batch_groups]
rankings = [collect(1:nrow(g)) for g in batch_groups]


using Random
using GLMakie

using Distributions

"""
The Gumbel distribution is used to model the extremes (max or min) of 
groups of samples from various distributions (wikipedia)
"""

include("/home/michael/3_labratory/makie_themes/everforest_makie.jl")

set_theme!(theme_everforest_dark())

underlying_dist = Normal(0, 2)

maxes = [maximum(rand(underlying_dist, 365)) for _ in 1:100]
mins  = [minimum(rand(underlying_dist, 365)) for _ in 1:100]

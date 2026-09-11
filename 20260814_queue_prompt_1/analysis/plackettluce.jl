using Random
using Distributions
using LogExpFunctions
using Bijectors
using Turing


struct PlackettLuce{T<:AbstractVector} <: DiscreteMultivariateDistribution
    η::T
end

Base.length(d::PlackettLuce) = length(d.η)


function Distributions.logpdf(d::PlackettLuce, π::AbstractVector{<:Integer})
    sum(d.η[π[k]] - logsumexp(@view d.η[π[k:end]]) for k in 1:length(π)-1)
end


function Distributions.rand(rng::AbstractRNG, d::PlackettLuce)
    sortperm(d.η .+ rand(rng, Gumbel(), length(d.η)), rev=true)
end


Bijectors.bijector(::PlackettLuce) = identity


function Distributions.fit_mle(::Type{<:PlackettLuce}, rankings; iters=1000)
    n = length(first(rankings))
    # 'position' order where each position corresponds to the item at that id.
    w = ones(n) 

    for _ in 1:iters
        denom = zeros(n)
        for π in rankings, k in 1:n-1
            rest = @view π[k:end]
            s = sum(@view w[rest])
            for i in rest
                denom[i] += 1 / s
            end
        end
        wins = zeros(n)
        for π in rankings, k in 1:n-1
            wins[π[k]] += 1
        end
        w = wins ./ denom
        w ./= sum(w) # Scale to sum to 1
    end
    return PlackettLuce(log.(w))
end


@model function plackettluce_regression(rankings, Xs)
    p = size(Xs, 2)
    β ~ MvNormal(zeros(p), 2.0 * I)
    η = Xs * β
    for i in eachindex(rankings)
        rankings[i] ~ PlackettLuce(η)
    end
end

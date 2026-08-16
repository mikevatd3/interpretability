using Random
using Distributions
using LogExpFunctions


struct PlackettLuce{T<:AbstractVector} <: DiscreteMultivariateDistribution
    """
    The differences in the elements of η matter, while their magnitude does not.

    When randomly generating, here is a table showing the distribution of Gumbel 
    nudge magnitude.

     -2.0  0.00456628
     -1.0  0.179374
      0.0  0.367879
      1.0  0.254646
      2.0  0.118205
      3.0  0.047369
      4.0  0.0179832
      5.0  0.0066927

    The mean for a Gumbel is at ~ 0.5772 (γ, the Euler-Mascheroni constant) and 
    95% of the mass is between -1.305 and 3.67.

    Also the difference between iid Gumbels is logistic (symmetric).
    """
    η::T
end


Base.length(d::PlackettLuce) = length(d.η)


function Distributions.logpdf(d::PlackettLuce, π::AbstractVector{<:Integer})
    sum(d.η[π[k]] - logsumexp(@view d.η[π[k:end]]) for k in 1:length(π)-1)
end


function Distributions.rand(rng::AbstractRNG, d::PlackettLuce)
    sortperm(d.η .+ rand(rng, Gumbel(), length(d.η)), rev=true)
end

using Distributions
using Turing

feature_cols = [
    :income_score,
    :loan_amount_score,
    :property_value_score,
    :logit_whi,
]

@model function pl_regression(rankings, Xts, p)
    β ~ MvNormal(zeros(p), 2.0 * I)

    for i in eachindex(rankings)
        η = Xts[i] * β
        rankings[i] ~ PlackettLuce(η)
    end
end

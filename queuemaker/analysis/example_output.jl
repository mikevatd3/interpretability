using DataFrames
using StatsBase
using Turing


income = [
    70000,
    73000,
    165000,
    9100000,
    139000,
    60000,
    219000,
    133000,
    152000,
    53000,
]

loan_amount = [
    35000.0 ,
    275000.0,
    145000.0,
    5005000.0,
    505000.0,
    265000.0,
    765000.0,
    385000.0,
    35000.0 ,
    135000.0,
]

property_value = [
    485000,
    652142,
    345000,
    12705000,
    1805000,
    355000,
    875000,
    465000,
    235000,
    652142,
]

ltv = loan_amount ./ property_value

X = hcat(income, loan_amount, property_value, ltv)

order_returned = [4,5,6,7,3,8,1,2,9,10]


@model function pl_regression(rankings, X)
    p = size(X, 2)
    β ~ MvNormal(zeros(p), 2.0 * I)
    η = X * β                        # log-worths per item

    for σ in rankings
        for k in 1:(length(σ) - 1)
            rest = σ[k:end]          # items not yet chosen
            Turing.@addlogprob! η[σ[k]] - logsumexp(η[rest])
        end
    end
end

chain = sample(pl_regression(order_returned, X), NUTS(), 1000)

using LibPQ
using DataFrames
using FlexiChains

include("plackettluce.jl")

conn = LibPQ.Connection("host=localhost dbname=michael user=michael")

result = execute(conn, """
SELECT user_index, array_agg(item_id + 1 order by rank_position)
FROM sushi_rankings_a
GROUP BY user_index;
"""; not_null=true)

rating_frame = DataFrame(result)

result = execute(conn, """
SELECT oiliness::FLOAT, eat_frequency::FLOAT, price::FLOAT
FROM sushi_items_b
WHERE item_id < 10;
"""; not_null=true)

features = DataFrame(result)

close(conn)

ratings = [Int.(rating) for rating in rating_frame[:, :array_agg]]
Xt = (features[:, [:oiliness, :eat_frequency, :price]] |> Matrix)
model = plackettluce_regression(ratings, Xt)
chain = sample(model, NUTS(), MCMCThreads(), 1000, 4)

mle_dist = fit(PlackettLuce, ratings)
best_order = sortperm(mle_dist.η, rev=true)

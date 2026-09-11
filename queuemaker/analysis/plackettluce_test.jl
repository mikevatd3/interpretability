include(joinpath(@__DIR__, "plackettluce.jl"))

using LibPQ
using DataFrames

conn = LibPQ.Connection("host=localhost dbname=michael user=michael")
result = execute(conn, """
SELECT user_index, array_agg(rank_position)
FROM (
    SELECT *
    FROM sushi_rankings_a
    ORDER BY user_index, item_id
) od
GROUP BY user_index;
""")
frame = DataFrame(result)
close(conn)


d = fit(PlackettLuce, frame[:, :array_agg])

A = []
y = []


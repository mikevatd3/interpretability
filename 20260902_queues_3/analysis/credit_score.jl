using GLMakie
using LibPQ
using DataFrames

using Distributions
using StatsBase

q = """
SELECT score_orig_r
FROM nsmo_puf;
"""

conn = LibPQ.Connection("dbname=hmda user=michael host=localhost")

result = execute(conn, q)
frame = DataFrame(result)

close(conn)

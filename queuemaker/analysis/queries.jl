using LibPQ
using DataFrames


function nsmo()
    conn = LibPQ.Connection("host=localhost dbname=hmda user=michael")
    result = execute(conn, """
    SELECT score_orig_r AS credit_score,
           loan_amount_cat AS loan_amount,
           x83 AS income,
           dti AS debt_to_income
    FROM nsmo_puf
    WHERE open_year = 2013
    AND score_orig_r IS NOT NULL
    AND x83 IS NOT NULL
    AND loan_amount_cat IS NOT NULL;
    """)
    frame = DataFrame(result)
    close(conn)
    return frame
end


function hmda_sample(batches=100)
    rows = batches * 10
    conn = LibPQ.Connection("host=localhost dbname=hmda user=michael")
    result = execute(conn, """
    SELECT hmda.census_tract AS tract_geoid,
           hmda.income::FLOAT * 1000 AS borrower_income,
           hmda.loan_amount::FLOAT,
           hmda.property_value::FLOAT,
           place.state_code AS property_state,
           place.place_name AS property_city
    FROM hmda_lar hmda
    JOIN hmda_tract_place_xwalk place
        ON hmda.census_tract = place.tract_geoid
    WHERE activity_year = 2024
    AND place_name IS NOT NULL
    AND income IS NOT NULL
    AND income NOT IN ('NA', 'Exempt')
    AND property_value NOT IN ('NA', 'Exempt')
    ORDER BY RANDOM()
    LIMIT \$1;
    """, [rows])
    frame = DataFrame(result)
    close(conn)
    return frame
end


function ohio_voter_names(batches=1000)
    rows = batches * 10
    conn = LibPQ.Connection("host=localhost dbname=michael user=michael")
    result = execute(conn, """
    SELECT last_name, first_name
    FROM oh_elections.voters
    ORDER BY RANDOM()
    LIMIT \$1;
    """, [rows])
    frame = DataFrame(result)
    close(conn)
    return frame
end


function rosenman_name_race(part::AbstractString)
    part in ("first", "last") || error("part must be \"first\" or \"last\"")
    conn = LibPQ.Connection("host=localhost dbname=hmda user=michael")
    result = execute(conn, """
    SELECT name, whi, bla, his, asi, oth
    FROM rosenman_$(part)_name_race;
    """)
    frame = DataFrame(result)
    close(conn)
    return frame
end


function hmda_tract_race(acs_year=2024)
    conn = LibPQ.Connection("host=localhost dbname=hmda user=michael")
    result = execute(conn, """
    SELECT tract_geoid, white, black, aian, asian, nhpi, other_race,
           two_or_more, hispanic
    FROM hmda_tract_race
    WHERE acs_year = \$1;
    """, [acs_year])
    frame = DataFrame(result)
    close(conn)
    return frame
end

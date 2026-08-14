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
    frame = dropmissing(frame)
    close(conn)
    return frame
end


function hmda_sample()
    conn = LibPQ.Connection("host=localhost dbname=hmda user=michael")
    result = execute(conn, """
    SELECT income::FLOAT * 1000 AS income,
           loan_amount::FLOAT
    FROM loan_sample_2025
    WHERE income <> 'NA'
    AND income::FLOAT > 0
    AND loan_amount <> 'NA'
    AND loan_amount::FLOAT > 0;
    """)
    frame = DataFrame(result)
    close(conn)
    return frame
end

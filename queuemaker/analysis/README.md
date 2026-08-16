# What do we need to know to generate realistic real-estate data

Generation itself lives in `../main.py` (Python) -- pulls real names from
`oh_elections.voters`, real loan records from `hmda_lar`, and adds BIFSG race
probabilities (`../bifsg.py`) using the Rosenman name-race tables and ACS
tract-race data. This directory (Julia) is for ad-hoc analysis of the
resulting distributions -- e.g. `plot_p_bla.jl`-style plots of a generated
run -- not for generating the dataset.

We want to generate loan candidates to order.

We can also generate building candidates to order as well.

## What fields are needed for the candidate level dataset

- Borrower Name
    - last_name, first_name
    - draw directly from voter files, or other sources then use BISG to come up
      with a racial probability estimation
- Borrower City
    - included in the estimation of BISG to estimate -- want this to be able to
      track if synthetic applicants are clearly following one racial pattern or
      ambiguous
- Income
    - This should be estimated ~ borrower city using ACS -- prob
- Loan Amount
    - Estimated from property city, income level, property value from 
- Property City
- Property Value
- Credit score
    - Drawn from a distribution


### Example

id | last_name  | first_name | income  | loan_amount | property_value | state_code | place_name     
---|------------|------------|---------|-------------|----------------|------------|------------
1  | BROWN      | RICHARD    |   70000 | 35000.0     | 485000         | MT         | Billings
2  | PHELAN     | LUKE       |   73000 | 275000.0    | NA             | GA         | Stonecrest
3  | FENDERSON  | KATHY      |  165000 | 145000.0    | 345000         | PA         | Campbelltown
4  | BISHOP     | DEAN       | 9100000 | 5005000.0   | 12705000       | UT         | Park City
5  | SAUNDERS   | REGINA     |  139000 | 505000.0    | 1805000        | NY         | Flower Hill
6  | SZILAGYI   | WILLIAM    |   60000 | 265000.0    | 355000         | NC         | Statesville
7  | BAKER      | MARK       |  219000 | 765000.0    | 875000         | KY         | Georgetown
8  | SAYLOR     | ANNA       |  133000 | 385000.0    | 465000         | OR         | Aloha
9  | STRICKLAND | ROBERT     |  152000 | 35000.0     | 235000         | OK         | Tulsa
10 | JOHNSON    | ANDERSON   |   53000 | 135000.0    | NA             | TX         | Mesquite


## What fields would be required for investment property recommendations

- Property City
- Property Value
- Property Neighborhood - MAYBE



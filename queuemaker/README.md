# Quemaker

## 2026-08-05

After last week, when I didn't have very good luck trying to extend the dialect
pairs work using SAEs and ablation, I started devising a new discrimination
measurement process. Also, I made sure to bring the work back to the housing
space.

I was talking to a person working on developing LLM tools at a mortgage company,
and they described a use case that I thought would be a perfect test for our
project. He described the use case like this "we'll give a list of mortgage
sales candidates to an LLM and have it look at the details and prioritize the
most likely leads to close, so the sales person knows who to call first and
spend the most time with."

I don't know that this will ultimately get implemented as described--maybe their
lawyers will get to them first. However, their idea seems like a fairly natural
use-case for LLMs in decision making that is likely governed by the Fair Housing
Act. The task also solves the problem from the prompt pairs work that I was
looking at before: while discriminatory terms are more likely when prompted with
AAVE dialect, those terms are still very unlikely. When you force the model to
order a set of candidates, any discriminatory behavior used in the decision can
be measured from actual model output.

I had another conversation with a person working on polling data in Michigan.
Since we're a non-Voting Rights Act state, his team uses a model to infer race
from the voter file based on name and address. I was thinking that this could be
sort of reverse engineered:

1. I found a reference dataset: "Race and Ethnicity Data for First, Middle, and
   Surnames" Rosenman, Evan T. R. and Olivella, Santiago and Imai, Kosuke (2023)
   which includes probabilities for each name component. It's a really nice
   dataset that includes both p(name|race) and p(race|name) for thousands of
   names.
2. Using the same logic as the person working on polling, we could also use
   census data about city demographics.

I'm still working through the details, but I think we can create a method to
generate many examples of the 'list of candidates' that have particular
structure -- the main loan details, loan amount and income, but then name and
address that is associated with our demographic measures from the other
datasets. This would have a nice effect of allowing the synthetic candidates
generated to have a fully continuous probability over 'latent' racial groups.

Then our final measure would be a regression analysis on the rankings,
(regression on rankings is a little tricky, but not too bad). We can then
control for things like debt to income, loan to value, etc and we don't need to
provide literal pairs within the candidate lists.

I think this will give us a rich data generation and model-output collection
setup that will help pin-point SAE features that could be related to
discriminatory effects.

## 2026-08-13

Hi Wonyoung,

That sounds good. I've been working mainly on the dataset generation process I
described last week.

To answer the questions from your previous email:

1. Yes, I've been building out the model so I can analyze model output to see if
   / how discriminatory effects show up. I see this as an initial measurement
   that can guide where to look. I think there are opportunities for looking at
   SAEs and the activation of the attention blocks 'per row & per field' of the
   synthetic input dataset. I like this because, if we can use a measurement
   like this with SAEs / attention activations to show why a behavior happens
   internally, nothing is stopping us from taking this 'outer measurement' on
   closed-weight models.

2. I've been working on the synthetic generation side and right now its a
   little bit clumsy. I'm following papers on the BISG and BIFSG technique for
   resolving race from name and place. (I also found a paper where they found
   modern LLMs to be more effective than these statistical models at accurately
   resolving race, so the idea that a model can do this task checks out). In my
   current draft, I pull full names at random from voter files and attach them
   to actual loan-level attributes from the Home Mortgage Disclosure act. I then
   use BIFSG to estimate from name and location the probability that the
   synthetic person is of a particular race. Unlike the analysts who are trying
   to figure out race on voter files, for this work, ambiguous race
   probabilities are useful because you can train the regression on the logit of
   that probability.  

I thought more about the causal part you mentioned more though. Originally I was
vaguely thinking causal a-la Donald Rubin, but the strategy could be to use
direct name-substituting identical pairs within otherwise the same candidate
dataset. This would be much a much clearer demonstration of the effect.

I also wanted to mention, I'm fully up and running on the GPU machine and
running experiments there. Everything is working great.

Talk soon,

Mike

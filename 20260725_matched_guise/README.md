# Matched guise probing

This is a logits method that looks at particular prompts and inspects the 
softmax output. Used in @hofmannAIGeneratesCovertly2024

The full 2019-pair AAVE/SAE dataset (beyond the one hardcoded example in
`matched_guise.py`) lives at `../../../data/aave_sae_pairs.tsv` — see
`../../../data/aave_sae_pairs.SOURCE.md` for where it came from and which
papers to cite if it's used.


## Issues that I have with the work

1. They have to log-scale and accumulate the bias from very low probabilities.
   These tokens are not likely to be predicted. Their ending calculation is
   essentially an average difference in log probabilities over the v

   It's a signal that the bias is
   encoded, but I don't think it's convincing on the claims that bias will be
   output by the model.

| word        |         aae |         sae |        log |
|:------------|------------:|------------:|-----------:|
| brilliant   | 5.30037e-05 | 6.59677e-05 | -0.218803  |
| dirty       | 8.89615e-06 | 3.72494e-06 |  0.870568  |
| intelligent | 6.45277e-05 | 6.7441e-05  | -0.0441585 |
| lazy        | 0.00125336  | 0.00044222  |  1.04178   |
| stupid      | 0.00082927  | 0.000304852 |  1.00072   |


## Main Idea

###  Experiments

1. [ ] Use the matched guise pairs with SAE
- Test the nested prompt
    - Within the parent prompt
    - Not in the parent prompt
- Try different layers

2. [ ] Some kind of average or difference in SAE activation across the pairs, 
       similar to their work?

3. [ ] Transfer the IDed nodes to our test prompt.
    - Try variations of the test to understand the activation patterns.

4. [X]  Check activated nodes against Neuronpedia's API for their label
   

### What I'm trying to learn

1. Are there reliable differences in activation across the pairs?
3. Do those activations develop at different layers?
3. How the parent prompt changes the SAE activations?
4. Are the identified activated notes transferable to our problem?

## Other Ideas

- [ ] TODO: Is there a way that we can accumulate tokens after the prompt and
      not just look at next prompt? We don't know what new tokens could wrench
      the prediction toward these biased outcomes if we only look at the next
      word.
- [ ] Check into the 'adjectives' and 'occupations' portion of the work and try
      to recreate that as well.


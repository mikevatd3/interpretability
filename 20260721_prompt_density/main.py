"""
Prompt density:

Provide a bunch of prompts that should share a next token, and look at how 
internally the representations differ.

- How much do the logits shift, even if the top match is the same? This is 
  kind of a question about the emptyness of high-dimensional spaces. (Is 'Paris' 
  always the same 'place'?)

- How does the model take these many paths to the same place?
"""

from transformer_lens import HookedTransformer
from dotenv import load_dotenv

load_dotenv()

model = HookedTransformer.from_pretrained("gpt2-small")

prompts = [ 
    "The capital city of France is",
    "The seat of France is",
    "The main city of France is",
    "The city which serves as seat of government of France is",
    "The most important city for the administration of France is",
    "The city in which leaders make decisions about France is",
    "The French city where all the government stuff is is",
    "If you're in France, and you want to meet the preseident of France, the city you should go to is",
]

for prompt in prompts:
    logits, cache = model.run_with_cache(prompt, prepend_bos=True)

    last_logits = logits[0, -1]
    top_k = last_logits.topk(3)
    
    word_list = [model.to_string(k).upper() for k in top_k.indices]

    print(f"{prompt} [{', '.join(word_list)}]")



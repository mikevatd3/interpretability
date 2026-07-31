import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Q:
    """Batching a causal LM's *last-token* prediction requires left-padding:
    with padding on the left, the real last token of every row in the batch
    always ends up at index -1, so logits[:, -1, :] is correct for the whole
    batch at once. (Right-padding would put a pad token at -1 for every
    sequence shorter than the batch's longest one.)
    """

    def __init__(
        self, tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel,
        outer_prompts: list[str], pairs: pd.DataFrame,
        batch_size: int = 32, device: str | None = None,
        cached_q: torch.Tensor | None = None,
    ) -> None:
        self.vocab_size = model.config.vocab_size
        self.pairs = pairs
        self.n_pairs = len(pairs)
        self.tokenizer = tokenizer
        self.model = model
        self.vs = outer_prompts
        self.batch_size = batch_size
        self.device = device or _get_device()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if cached_q is not None:
            self.cached_q = cached_q
            return

        self.model.to(self.device)
        self.cached_q = self.evaluate_pairs()

    def _last_token_probs(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(self.device)

        # GPT-2 doesn't infer position_ids from attention_mask the way it
        # does with attention itself -- left-padded rows need this or the
        # real tokens silently get the wrong position embeddings. Same fix
        # HF's GenerationMixin applies for left-padded batches:
        # https://github.com/huggingface/transformers/blob/bd970fda476f204f9852fb551aa07a4b3fbc5540/src/transformers/generation/utils.py#L734-L737
        # Background on why left-padding is required at all:
        # https://huggingface.co/docs/transformers/en/llm_tutorial#padding-side
        attention_mask = encoded["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        output = self.model(
            input_ids=encoded["input_ids"],
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        return F.softmax(output.logits[:, -1, :], dim=-1)

    def evaluate_pairs(self) -> torch.Tensor:
        accumulation = torch.zeros(self.vocab_size, device=self.device)

        self.model.eval()

        # Flatten (template x pair) into one list so batches can cut across
        # template/pair boundaries instead of being capped at n_pairs each.
        aave_texts = [
            v.format(t=pair["aave"])
            for v in self.vs
            for _, pair in self.pairs.iterrows()
        ]
        sae_texts = [
            v.format(t=pair["sae"])
            for v in self.vs
            for _, pair in self.pairs.iterrows()
        ]
        n_total = len(aave_texts)

        with torch.no_grad():
            for start in tqdm(
                range(0, n_total, self.batch_size),
                desc="Batches",
                total=(n_total + self.batch_size - 1) // self.batch_size,
            ):
                end = start + self.batch_size

                aave_probs = self._last_token_probs(aave_texts[start:end])
                sae_probs = self._last_token_probs(sae_texts[start:end])

                accumulation += torch.log(aave_probs / sae_probs).sum(dim=0)

        return (accumulation / (self.n_pairs * len(self.vs))).cpu()

    def __call__(self, token: str) -> float:
        i = self.tokenizer.encode(token)[0]  # Grab the first

        return self.cached_q[i].item()

    def save(self, path) -> None:
        torch.save(self.cached_q, path)

    @classmethod
    def load(cls, tokenizer, model, outer_prompts, pairs, path, **kwargs):
        cached_q = torch.load(path)
        return cls(tokenizer, model, outer_prompts, pairs, cached_q=cached_q, **kwargs)

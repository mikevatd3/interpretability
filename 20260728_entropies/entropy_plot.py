from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import get_device
from entropy import sentence_entropy
from plot import plot_entropy


load_dotenv()


EXAMPLE_SENTENCE = "The shots which killed President Kennedy and wounded Governor Connally were fired from the sixth-floor window at the southeast corner of the Texas School Book Depository."


def main(sentence: str = EXAMPLE_SENTENCE) -> None:
    device = get_device()
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    df = sentence_entropy(sentence, model, tokenizer, device=device)
    print(df)

    # "the model chose this word from this tight of a band of choices"
    plot_entropy(df, column="entropy_before_bits", title=f'Next-token entropy: "{sentence}"').show()


if __name__ == "__main__":
    main()

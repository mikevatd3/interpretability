import psycopg
from pgvector.psycopg import register_vector
from transformer_lens.model_bridge import TransformerBridge

DEVICE = "cpu"

TABLE = "gpt2_vocab"

DSN = "host=localhost dbname=michael user=michael"


def load_vocab_and_embeddings():
    model = TransformerBridge.boot_transformers("gpt2", device=DEVICE)

    vocab_size, d_model = model.W_E.shape
    names = [
        model.tokenizer.decode([i]).replace("\x00", "")
        for i in range(vocab_size)
    ]
    vecs = model.W_E.detach().cpu().numpy()

    return names, vecs, d_model


def main():
    names, vecs, d_model = load_vocab_and_embeddings()

    conn = psycopg.connect(DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    conn.execute(f"""
        CREATE TABLE {TABLE} (
            token_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            vec VECTOR({d_model}) NOT NULL
        )
    """)

    with conn.cursor() as cur:
        with cur.copy(f"COPY {TABLE} (token_id, name, vec) FROM STDIN WITH (FORMAT binary)") as copy:
            copy.set_types(["int4", "text", "vector"])
            for token_id, (name, vec) in enumerate(zip(names, vecs)):
                copy.write_row((token_id, name, vec))

    print(f"Loaded {len(names)} rows into {TABLE} (dim={d_model})")
    conn.close()


if __name__ == "__main__":
    main()

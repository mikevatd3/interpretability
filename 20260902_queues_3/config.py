import os
import torch
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


load_dotenv()


def get_device() -> str:
    """cuda > mps > cpu, whichever is actually available in this environment."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_db(name: str) -> Engine:
    """Build a SQLAlchemy engine for the database `name`, using host/port/
    user/password from the environment (DB_HOST, DB_PORT, DB_USER,
    DB_PASSWORD)."""
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")

    url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"
    return create_engine(url)


DEVICE = get_device()

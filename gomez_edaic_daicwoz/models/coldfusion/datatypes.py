from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int
    embedding_dim: int
    hidden_size: int
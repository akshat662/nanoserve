from dataclasses import dataclass

import torch
import yaml

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class ServerConfig:
    model_id: str
    device: str
    dtype: torch.dtype
    attn_implementation: str
    max_slots: int
    max_seq_len: int
    max_new_tokens: int
    batch_token_budget: int


def load_config(path: str = "configs/default.yaml") -> ServerConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    device = raw["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = _DTYPE_MAP[raw["dtype"]]

    return ServerConfig(
        model_id=raw["model_id"],
        device=device,
        dtype=dtype,
        attn_implementation=raw["attn_implementation"],
        max_slots=raw["max_slots"],
        max_seq_len=raw["max_seq_len"],
        max_new_tokens=raw["max_new_tokens"],
        batch_token_budget=raw["batch_token_budget"],
    )

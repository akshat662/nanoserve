"""Print Qwen2.5 config fields and KV cache size estimates. Loads config only, not weights."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import transformers
from transformers import AutoConfig

from server.config import load_config

DTYPE_BYTES = {"float16": 2, "float32": 4}


def main() -> None:
    server_config = load_config()
    config = AutoConfig.from_pretrained(server_config.model_id)

    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    hidden_size = config.hidden_size
    max_position_embeddings = config.max_position_embeddings
    eos_token_id = config.eos_token_id

    head_dim = getattr(config, "head_dim", None)
    head_dim_note = ""
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
        head_dim_note = " (derived as hidden_size // num_attention_heads; absent from config)"

    print(f"transformers.__version__: {transformers.__version__}")
    print(f"num_hidden_layers: {num_hidden_layers}")
    print(f"num_attention_heads: {num_attention_heads}")
    print(f"num_key_value_heads: {num_key_value_heads}")
    print(f"head_dim: {head_dim}{head_dim_note}")
    print(f"hidden_size: {hidden_size}")
    print(f"max_position_embeddings: {max_position_embeddings}")
    print(f"eos_token_id: {eos_token_id}")

    for dtype_name, dtype_bytes in DTYPE_BYTES.items():
        bytes_per_token = 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes
        print(f"KV cache bytes/token ({dtype_name}): {bytes_per_token}")

    total_tokens = server_config.max_slots * server_config.max_seq_len
    for dtype_name, dtype_bytes in DTYPE_BYTES.items():
        bytes_per_token = 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes
        total_mb = bytes_per_token * total_tokens / (1024 * 1024)
        print(
            f"Total cache size ({dtype_name}, max_slots={server_config.max_slots} x "
            f"max_seq_len={server_config.max_seq_len}): {total_mb:.2f} MB"
        )


if __name__ == "__main__":
    main()

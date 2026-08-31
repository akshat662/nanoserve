"""Points the official OpenAI python client at a running nanoserve instance.
Nothing else changes from a real OpenAI call — that is the demo.

Start the server first: make serve
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    max_tokens=32,
)

print(response.choices[0].message.content)

"""Verification for the FastAPI server. Test (f) is the one that catches a
blocking call in the handler — if step() ever runs on the event loop directly
instead of via asyncio.to_thread, concurrent requests serialize or hang.
"""

import asyncio
import sys
import time
from pathlib import Path

import httpx
from openai.types.chat import ChatCompletion

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.types import Request  # noqa: E402

MAX_TOKENS = 8  # keep tests fast on CPU per the suite's ~2 minute budget


def test_chat_completions_returns_valid_openai_shape(app_client):
    resp = app_client.post(
        "/v1/chat/completions",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "Hi."}], "max_tokens": MAX_TOKENS},
    )
    assert resp.status_code == 200
    body = resp.json()

    # the official openai client's own Pydantic model must accept this body verbatim
    parsed = ChatCompletion.model_validate(body)
    assert parsed.object == "chat.completion"
    assert parsed.id.startswith("chatcmpl-")
    assert len(parsed.choices[0].message.content) > 0
    assert parsed.choices[0].finish_reason in ("stop", "length")
    assert parsed.usage.total_tokens == parsed.usage.prompt_tokens + parsed.usage.completion_tokens


def test_stream_true_returns_400(app_client):
    resp = app_client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "Hi."}], "stream": True},
    )
    assert resp.status_code == 400


def test_usage_matches_actual_token_lengths(app_client, shared_engine):
    messages = [{"role": "user", "content": "What is 2+2?"}]
    resp = app_client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": messages, "max_tokens": MAX_TOKENS},
    )
    usage = resp.json()["usage"]

    expected_prompt_ids = shared_engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]
    req = Request(request_id="verify-g", prompt_token_ids=expected_prompt_ids, max_new_tokens=MAX_TOKENS, arrival_time=time.time())
    expected_seq = shared_engine.generate([req])[0]

    assert usage["prompt_tokens"] == len(expected_prompt_ids)
    assert usage["completion_tokens"] == len(expected_seq.output_token_ids)
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_concurrent_requests_do_not_hang(shared_engine):
    from server.app import app, lifespan

    app.state.config = shared_engine.config
    app.state.engine = shared_engine

    prompts = ["Hi.", "Hello!", "Yes."]
    payloads = [{"model": "x", "messages": [{"role": "user", "content": p}], "max_tokens": MAX_TOKENS} for p in prompts]

    async def run() -> list[httpx.Response]:
        async with lifespan(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                tasks = [client.post("/v1/chat/completions", json=payload) for payload in payloads]
                return await asyncio.gather(*tasks)

    responses = asyncio.run(asyncio.wait_for(run(), timeout=90))

    assert len(responses) == 3
    for resp in responses:
        assert resp.status_code == 200
        assert len(resp.json()["choices"][0]["message"]["content"]) > 0


def test_health_and_models(app_client):
    health = app_client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert isinstance(health.json()["free_slots"], int)

    models = app_client.get("/v1/models")
    assert models.status_code == 200
    body = models.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "Qwen/Qwen2.5-0.5B-Instruct"

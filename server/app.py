"""FastAPI server: one background task owns the engine and drains a queue of
(Request, Future) pairs by calling step() in a thread. Handlers never touch
the model directly — that is what keeps concurrent requests from blocking on
each other or on the event loop.
"""

import asyncio
import time
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi import Request as HTTPRequest
from pydantic import BaseModel

from server.config import ServerConfig, load_config
from server.engine import Engine
from server.scheduler import ContinuousBatchScheduler
from server.static_batch import StaticBatchEngine
from server.types import Request as EngineRequest


def build_engine(config: ServerConfig):
    base = Engine(config)
    if config.engine == "static":
        return StaticBatchEngine(base)
    if config.engine == "scheduler":
        return ContinuousBatchScheduler(base)
    return base


async def _run_loop(app: FastAPI) -> None:
    engine, queue, pending = app.state.engine, app.state.queue, app.state.pending_futures
    while True:
        while not queue.empty():
            req, future = queue.get_nowait()
            engine.submit(req)
            pending[req.request_id] = future

        if engine.has_work():
            finished = await asyncio.to_thread(engine.step)
            for seq in finished:
                future = pending.pop(seq.request_id, None)
                if future is not None and not future.done():
                    future.set_result(seq)
        else:
            await asyncio.sleep(0.005)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "config"):
        app.state.config = load_config()
    if not hasattr(app.state, "engine"):
        app.state.engine = build_engine(app.state.config)
    app.state.queue = asyncio.Queue()
    app.state.pending_futures = {}
    task = asyncio.create_task(_run_loop(app))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    stream: bool = False


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, http_request: HTTPRequest):
    if payload.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported; SSE streaming is a V1 non-goal")

    state = http_request.app.state
    engine, config = state.engine, state.config

    messages = [{"role": m.role, "content": m.content} for m in payload.messages]
    prompt_ids = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]
    max_new_tokens = payload.max_tokens or config.max_new_tokens

    request_id = f"chatcmpl-{uuid4().hex}"
    req = EngineRequest(
        request_id=request_id, prompt_token_ids=prompt_ids, max_new_tokens=max_new_tokens, arrival_time=time.time()
    )

    future = asyncio.get_running_loop().create_future()
    await state.queue.put((req, future))
    seq = await future

    content = engine.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
    prompt_tokens, completion_tokens = len(seq.prompt_token_ids), len(seq.output_token_ids)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": config.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop" if seq.finish_reason == "stop" else "length",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/health")
async def health(http_request: HTTPRequest):
    return {"status": "ok", "free_slots": http_request.app.state.engine.cache.num_free_slots()}


@app.get("/v1/models")
async def list_models(http_request: HTTPRequest):
    model_id = http_request.app.state.config.model_id
    return {"object": "list", "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "nanoserve"}]}

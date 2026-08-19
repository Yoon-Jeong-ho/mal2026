"""OpenAI-compatible HTTP boundary required by the MAL2026 evaluator."""
from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from .contracts import compact_participant_json, extract_prompt_essay
from .pipeline import Completion, Pipeline, load_pipeline


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stop: str | list[str] | None = None


def _message_dicts(request: ChatCompletionRequest) -> list[dict[str, Any]]:
    return [message.model_dump() for message in request.messages]


def create_app(
    *,
    pipeline: Pipeline | None = None,
    pipeline_factory: Callable[[], Pipeline] = load_pipeline,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        app.state.load_error = None
        app.state.pipeline = pipeline
        app.state.inference_lock = asyncio.Lock()
        try:
            if app.state.pipeline is None:
                app.state.pipeline = await asyncio.to_thread(pipeline_factory)
            app.state.ready = True
        except Exception as exc:  # health remains 503 and startup evidence stays in stderr
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            print(f"MAL2026 pipeline startup failed: {app.state.load_error}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
        yield

    app = FastAPI(title="MAL2026 writing scorer", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health(response: Response) -> dict[str, str]:
        if not getattr(app.state, "ready", False):
            response.status_code = 503
            return {"status": "loading" if app.state.load_error is None else "error"}
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="model is not ready")
        model_name = app.state.pipeline.served_model_name
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "created": 0, "owned_by": "mal2026"}],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: ChatCompletionRequest) -> dict[str, Any]:
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="model is not ready")
        active: Pipeline = app.state.pipeline
        if request.model != active.served_model_name:
            raise HTTPException(status_code=404, detail="unknown model")
        try:
            messages = _message_dicts(request)
            is_task_request = extract_prompt_essay(messages) is not None
            async with app.state.inference_lock:
                completion: Completion = await asyncio.to_thread(
                    active.complete,
                    messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    seed=request.seed,
                    stop=request.stop,
                )
            if is_task_request:
                # Enforce one compact top-level participant object at the HTTP
                # boundary even if a future pipeline implementation regresses.
                completion = Completion(
                    content=compact_participant_json(completion.content),
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"inference failed: {type(exc).__name__}") from exc
        if not isinstance(completion.content, str) or not completion.content.strip():
            raise HTTPException(status_code=500, detail="inference returned blank content")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": active.served_model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": completion.content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": int(completion.prompt_tokens),
                "completion_tokens": int(completion.completion_tokens),
                "total_tokens": int(completion.prompt_tokens + completion.completion_tokens),
            },
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "mal2026_submission.server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level=os.environ.get("MAL2026_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()

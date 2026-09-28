"""HTTP boundary: authentication, validation, bounded execution and safe errors."""
import asyncio
from collections import deque
from contextlib import asynccontextmanager
import os
import secrets
from time import monotonic
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.agent import AgentFailure, run_agent


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str]


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        key = os.getenv("OPENAI_API_KEY", "").strip()
        app.state.client = AsyncOpenAI(api_key=key, timeout=20, max_retries=0) if key else None
        app.state.model = os.getenv("OPENAI_MODEL", "").strip()
        app.state.token = os.getenv("APP_API_TOKEN", "").strip()
        app.state.requests = deque()
        app.state.lock = asyncio.Lock()
        app.state.slots = asyncio.Semaphore(2)
        yield
        if app.state.client is not None:
            await app.state.client.close()

    app = FastAPI(title="FastAPI + LLM Agent", version="0.1.0", lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)

    async def authorize(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> None:
        expected = app.state.token
        if not expected or not app.state.model or app.state.client is None:
            raise HTTPException(503, "Configure APP_API_TOKEN, OPENAI_API_KEY and OPENAI_MODEL on the server.")
        if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), expected.encode()):
            raise HTTPException(401, "Invalid application token", headers={"WWW-Authenticate": "Bearer"})
        async with app.state.lock:
            now = monotonic()
            while app.state.requests and app.state.requests[0] <= now - 60:
                app.state.requests.popleft()
            if len(app.state.requests) >= 10:
                raise HTTPException(429, "Request limit reached", headers={"Retry-After": "60"})
            app.state.requests.append(now)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(authorize)])
    async def chat(body: ChatRequest) -> ChatResponse:
        try:
            async with asyncio.timeout(45):
                async with app.state.slots:
                    answer, used = await run_agent(app.state.client, app.state.model, body.message)
            return ChatResponse(answer=answer, tools_used=used)
        except TimeoutError:
            raise HTTPException(504, "The agent timed out.") from None
        except (APIError, AgentFailure):
            raise HTTPException(502, "The agent could not complete this request.") from None

    return app


app = create_app()

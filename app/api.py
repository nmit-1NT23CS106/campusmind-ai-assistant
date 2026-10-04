"""HTTP API. Run: uvicorn app.api:app --port 8000

NOTE: role and student_id come from the request body. That is fine for a demo with fake data,
but a real deployment must take identity from the university's login (SSO/JWT), never from the client.
API_TOKEN and ADMIN_TOKEN add a shared-secret check so a public demo can't be abused.
"""
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .llm import LLMError
from .services import build_state
from .sql_tool import Scope
from .telemetry import log_interaction, summarize


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    role: Literal["student", "admin"] = "student"
    student_id: Optional[int] = None


class RateLimiter:
    def __init__(self, per_minute):
        self.per_minute, self.hits, self.lock = per_minute, defaultdict(deque), threading.Lock()

    def allow(self, key):
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            window = self.hits[key]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.per_minute:
                return False
            window.append(now)
            return True


def _same(a, b):
    return secrets.compare_digest((a or "").encode(), (b or "").encode())


def create_app(state=None):
    state = state or build_state()
    settings = state.settings
    limiter = RateLimiter(settings.rate_limit_per_min)
    app = FastAPI(title="CampusMind", version="1.0.0")

    def guard(request, api_token):
        if not limiter.allow(request.client.host if request.client else "unknown"):
            raise HTTPException(429, "Too many requests. Please wait a minute.")
        if settings.api_token and not _same(api_token, settings.api_token):
            raise HTTPException(401, "Missing or invalid X-API-Token.")

    @app.get("/health")
    def health():
        return {"status": "ok", "retriever": state.retriever_name, "llm_configured": state.llm_ready,
                "warning": state.warning, **summarize(settings.log_path)}

    @app.get("/demo/students")
    def demo_students(request: Request, limit: int = 5, x_api_token: Optional[str] = Header(None)):
        guard(request, x_api_token)
        return state.sql_tool.demo_students(min(max(limit, 1), 20))

    @app.post("/ask")
    def ask(body: AskRequest, request: Request, x_api_token: Optional[str] = Header(None),
            x_admin_token: Optional[str] = Header(None)):
        guard(request, x_api_token)
        if body.role == "admin":
            if settings.admin_token and not _same(x_admin_token, settings.admin_token):
                raise HTTPException(403, "Admin access needs a valid X-Admin-Token.")
            scope = Scope("admin")
        else:
            if body.student_id is None:
                raise HTTPException(422, "student_id is required for role 'student'.")
            if not state.sql_tool.student_exists(body.student_id):
                raise HTTPException(404, "No such student.")
            scope = Scope("student", body.student_id)

        if not state.llm_ready:
            raise HTTPException(503, "The language model is not configured. Set LLM_PROVIDER and an API key in .env.")

        try:
            result = state.agent().ask(body.question, scope)
        except LLMError as exc:
            log_interaction(settings.log_path, question=body.question, role=body.role,
                            student_id=body.student_id, error=f"llm: {exc}")
            raise HTTPException(502, "The language model request failed. Please try again.")
        except Exception as exc:
            log_interaction(settings.log_path, question=body.question, role=body.role,
                            student_id=body.student_id, error=type(exc).__name__)
            raise HTTPException(500, "Something went wrong while answering.")

        log_interaction(settings.log_path, question=body.question, role=body.role,
                        student_id=body.student_id, result=result)
        return {**result.to_dict(), "role": body.role, "retriever": state.retriever_name}

    return app


def __getattr__(name):  # `uvicorn app.api:app` builds the app lazily, so importing this module stays cheap
    if name == "app":
        global _app
        try:
            return _app
        except NameError:
            _app = create_app()
            return _app
    raise AttributeError(name)

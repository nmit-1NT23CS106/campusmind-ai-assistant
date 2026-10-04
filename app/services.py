"""Wire the pieces together from environment settings."""
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .chunking import load_chunks
from .llm import get_llm, llm_configured
from .retrieval import build_retriever
from .sql_tool import SQLTool

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("campusmind")


@dataclass
class Settings:
    db_path: Path = ROOT / "data" / "campus.db"
    docs_dir: Path = ROOT / "data" / "docs"
    log_path: Path = ROOT / "data" / "logs" / "interactions.jsonl"
    retriever: str = "hybrid"
    api_token: str = ""      # if set, every request needs header X-API-Token
    admin_token: str = ""    # if set, role=admin additionally needs header X-Admin-Token
    rate_limit_per_min: int = 20

    @classmethod
    def from_env(cls):
        load_dotenv()
        env = os.environ
        return cls(
            db_path=Path(env.get("DB_PATH", cls.db_path)), docs_dir=Path(env.get("DOCS_DIR", cls.docs_dir)),
            log_path=Path(env.get("LOG_PATH", cls.log_path)), retriever=env.get("RETRIEVER", "hybrid"),
            api_token=env.get("API_TOKEN", ""), admin_token=env.get("ADMIN_TOKEN", ""),
            rate_limit_per_min=int(env.get("RATE_LIMIT_PER_MIN", "20")),
        )


class AppState:
    def __init__(self, settings, retriever, sql_tool, retriever_name, llm=None, warning=None):
        self.settings, self.retriever, self.sql_tool = settings, retriever, sql_tool
        self.retriever_name, self.warning = retriever_name, warning
        self._llm, self._agent, self._lock = llm, None, threading.Lock()

    @property
    def llm_ready(self):
        return self._llm is not None or llm_configured()

    def agent(self):
        with self._lock:
            if self._agent is None:
                self._agent = Agent(self._llm or get_llm(), self.retriever, self.sql_tool)
            return self._agent


def build_state(settings=None, llm=None, retriever=None):
    settings = settings or Settings.from_env()
    if not settings.db_path.exists():
        from data.seed import build
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        build(settings.db_path)

    warning, name = None, settings.retriever
    if retriever is None:
        chunks = load_chunks(settings.docs_dir)
        try:
            retriever = build_retriever(name, chunks)
        except Exception as exc:  # e.g. the embedding model could not be downloaded
            if name == "bm25":
                raise
            warning = f"Could not start '{name}' retrieval ({type(exc).__name__}); using bm25 instead."
            log.warning(warning)
            retriever, name = build_retriever("bm25", chunks), "bm25"
    return AppState(settings, retriever, SQLTool(settings.db_path), name, llm=llm, warning=warning)

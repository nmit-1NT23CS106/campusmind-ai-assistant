from dotenv import dotenv_values

from app.services import Settings, clean_env_value
from tests.conftest import ROOT


def test_env_example_has_no_values_that_are_really_comments():
    """python-dotenv reads 'KEY=   # note' as KEY='# note'. Keep comments on their own lines."""
    for key, value in dotenv_values(ROOT / ".env.example").items():
        assert not (value or "").startswith("#"), f"{key} would be read as a comment string"


def test_env_example_defaults_are_safe_for_local_use():
    v = dotenv_values(ROOT / ".env.example")
    assert not v["API_TOKEN"] and not v["ADMIN_TOKEN"]
    assert v["LLM_PROVIDER"] in ("anthropic", "openai", "groq", "gemini")


def test_clean_env_value():
    assert clean_env_value(None) == "" and clean_env_value("  ") == ""
    assert clean_env_value("# if set, every request needs a header") == ""
    assert clean_env_value("  secret  ") == "secret"


def test_stray_comment_values_do_not_turn_on_auth(monkeypatch):
    monkeypatch.setattr("app.services.load_dotenv", lambda: None)
    monkeypatch.setenv("API_TOKEN", "# if set, every request needs header X-API-Token")
    monkeypatch.setenv("ADMIN_TOKEN", "# if set, role=admin also needs header")
    s = Settings.from_env()
    assert s.api_token == "" and s.admin_token == ""

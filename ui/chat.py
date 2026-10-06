"""Streamlit chat page. Run: streamlit run ui/chat.py   (the API must be running)"""
import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

STUDENT_EXAMPLES = [
    "What is the minimum attendance to sit the exams?",
    "What is my attendance in each course?",
    "Am I eligible to write the end-semester exams?",
    "How much does revaluation cost?",
]
ADMIN_EXAMPLES = [
    "Which students have attendance below 75% in CS301?",
    "Which course has the lowest average attendance?",
    "Show the top 5 students by CGPA",
    "What is the punishment for malpractice in exams?",
]


def api_headers(api_token, admin_token=""):
    headers = {}
    if api_token:
        headers["X-API-Token"] = api_token
    if admin_token:
        headers["X-Admin-Token"] = admin_token
    return headers


def call_api(payload, api_token="", admin_token="", base_url=API_URL):
    """Return (ok, data). On failure data is a short message fit to show the user."""
    try:
        resp = requests.post(f"{base_url}/ask", json=payload, headers=api_headers(api_token, admin_token), timeout=120)
    except requests.RequestException:
        return False, f"Can't reach the API at {base_url}. Is it running?"
    if resp.status_code == 200:
        return True, resp.json()
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    return False, f"{resp.status_code}: {detail}"


def get_json(path, api_token="", base_url=API_URL):
    try:
        resp = requests.get(f"{base_url}{path}", headers=api_headers(api_token), timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None


def show_details(meta):
    with st.expander("How I answered"):
        st.caption(f"Tools: {', '.join(meta['tools_used']) or 'none'}  |  {meta['steps']} step(s)  |  "
                   f"{meta['latency_ms'] / 1000:.1f}s  |  {meta['input_tokens']} in / {meta['output_tokens']} out tokens  |  "
                   f"retrieval: {meta.get('retriever', '?')}")
        for run in meta["sql_runs"]:
            st.code(run["sql"] or "(no SQL)", language="sql")
            if run["error"]:
                st.error(run["error"])
            elif run["rows"]:
                st.dataframe(pd.DataFrame(run["rows"], columns=run["columns"]), hide_index=True)
                if run["truncated"]:
                    st.caption("Result truncated at the row limit.")
            else:
                st.caption("No rows returned.")
        for step in meta["trace"]:
            if step["action"] in ("invalid_reply", "answer_rejected", "unknown_action", "step_limit"):
                st.warning(f"Step {step['step']}: {step['action']}: {step.get('detail', '')}")
            if step["action"] == "search_docs":
                st.markdown(f"**Searched documents:** {step['detail']}")
                for cite in step.get("retrieved", []):
                    st.caption(f"- {cite}")


def render(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        meta = message.get("meta")
        if meta:
            if meta["sources"]:
                st.caption("Sources: " + "; ".join(meta["sources"]))
            show_details(meta)


st.set_page_config(page_title="CampusMind", page_icon="🎓", layout="wide")
st.title("🎓 CampusMind")
st.caption("Ask about university policies and student data. Demo with fictional data.")

with st.sidebar:
    st.header("Sign in (demo)")
    api_token = st.text_input("API token", type="password", value=("" if os.environ.get("API_TOKEN", "").strip().startswith("#") else os.environ.get("API_TOKEN", "").strip()))
    role = st.radio("I am a", ["Student", "Admin"]).lower()
    student_id, admin_token = None, ""
    if role == "student":
        students = get_json("/demo/students?limit=8", api_token) or []
        if students:
            labels = {f"{s['name']} ({s['student_id']})": s["student_id"] for s in students}
            student_id = labels[st.selectbox("Student", list(labels))]
        else:
            student_id = int(st.number_input("Student ID", value=20260001, step=1))
    else:
        admin_token = st.text_input("Admin token", type="password")
    health = get_json("/health")
    if health:
        st.success(f"API online | retrieval: {health['retriever']}")
        if not health["llm_configured"]:
            st.warning("No LLM key configured on the server.")
        if health.get("warning"):
            st.info(health["warning"])
    else:
        st.error("API offline")
    st.caption("Students can only see their own records. Admins can query everyone.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    render(m)

st.write("Try one:")
columns = st.columns(2)
examples = STUDENT_EXAMPLES if role == "student" else ADMIN_EXAMPLES
clicked = None
for i, example in enumerate(examples):
    if columns[i % 2].button(example, key=f"ex{i}"):
        clicked = example

prompt = st.chat_input("Ask a question") or clicked
if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render(user_msg)
    with st.spinner("Thinking..."):
        ok, data = call_api({"question": prompt, "role": role, "student_id": student_id}, api_token, admin_token)
    reply = ({"role": "assistant", "content": data["answer"], "meta": data} if ok
             else {"role": "assistant", "content": f"Sorry, that failed. {data}"})
    st.session_state.messages.append(reply)
    render(reply)

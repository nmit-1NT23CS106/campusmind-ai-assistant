# Starts the API in a new window, then the chat UI here.  Run:  .\start.ps1
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\.venv\Scripts\python -m uvicorn app.api:app --port 8000"
Start-Sleep -Seconds 8
.\.venv\Scripts\python -m streamlit run ui/chat.py

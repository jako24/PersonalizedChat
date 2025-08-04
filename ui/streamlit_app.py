import os
import sys
import requests
import streamlit as st
from uuid import uuid4
import uvicorn
from threading import Thread

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.server import app as fastapi_app

# --- Backend Server --- 

def run_backend():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

backend_thread = Thread(target=run_backend)
backend_thread.daemon = True
backend_thread.start()

# --- Streamlit App ---

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")
st.set_page_config(page_title="Product Chatbot", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

st.title("🛒 Product Chatbot (Ingredients Only)")

with st.sidebar:
    st.markdown("**Session ID**")
    st.code(st.session_state.session_id, language="text")
    st.markdown("Backend:")
    st.text_input("URL", value=BACKEND_URL, key="backend_url")

prompt = st.chat_input("Escribe tu pregunta (ej. “¿Cuáles son los beneficios de ZARZAPARRILLA?”)")
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {"session_id": st.session_state.session_id, "message": prompt}
    try:
        r = requests.post(st.session_state.backend_url, json=payload, timeout=60)
        # st.code(f"Status code: {r.status_code}")
        # st.code(f"Response text: {r.text}")
        r.raise_for_status()
        reply = r.json()["reply"]
    except Exception as e:
        reply = f"Error: {e}"

    st.session_state.history.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)
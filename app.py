from groq import Groq
import numpy as np
import cv2
import streamlit as st
import uuid
from datetime import datetime
import time
import random
import os
import sys
import requests
import json
import speech_recognition as sr
from PIL import Image
import pytesseract
from dotenv import load_dotenv

load_dotenv()


# OCR Setup
try:
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
    if sys.platform.startswith('win'):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            st.error("Tesseract not found at C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
            OCR_AVAILABLE = False
except ImportError:
    OCR_AVAILABLE = False


# Page config
st.set_page_config(
    page_title="Code Gen Ai",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Base session ----------
if "settings" not in st.session_state:
    st.session_state.settings = {
        "model": "llama-3.1-8b-instant",
        "temperature": 0.7,
        "font_size": "Medium",
        "particles": False,
        "user_name": None,
        "role": None,
        "ai_name": "Code Gen Ai",
        "theme": "Dark",
    }


# ---------- Mode prompts ----------
BASE_MODE_PROMPTS = {
    "Debug code": "You are a senior debugging assistant. Find and fix bugs in this code.",
    "Solve problem": "You are a competitive programming expert. Solve the following problem with explanation and code.",
    "Explain code": "You are a teacher. Explain what this code does step by step in simple language.",
    "Practise code": "You are a coding coach. Give small practice tasks and solutions based on the topic.",
    "Learn new technology": "You are a technology mentor. Explain and guide the user to learn new tools, frameworks, or technologies with practical examples.",
}


def get_modes_for_role(role: str):
    role = (role or "").lower()
    if role in ["student", "teacher", "coder"]:
        return ["Debug code", "Solve problem", "Explain code", "Practise code"]
    if role in ["employee", "business"]:
        return ["Debug code", "Learn new technology", "Explain code", "Practise code"]
    return ["Debug code", "Solve problem", "Explain code", "Practise code"]


# ---------- Theme Configurations ----------
def get_theme_colors(theme):
    if theme == "Dark":
        return {
            "bg_gradient_1": "#0f172a",
            "bg_gradient_2": "#1e293b",
            "bg_gradient_3": "#020617",
            "text_primary": "#f8fafc",
            "text_secondary": "#94a3b8",
            "card_bg": "rgba(30, 41, 59, 0.7)",
            "card_border": "rgba(148, 163, 184, 0.1)",
            "sidebar_bg": "rgba(15, 23, 42, 0.95)",
            "user_bubble": "#2563eb",
            "user_text": "#ffffff",
            "assistant_bubble": "rgba(30, 41, 59, 0.7)",
            "assistant_text": "#f1f5f9",
            "accent": "#3b82f6",
            "accent_glow": "rgba(59, 130, 246, 0.5)",
            "input_bg": "rgba(15, 23, 42, 0.6)",
            "input_border": "rgba(59, 130, 246, 0.3)",
            "shadow": "0 3px 4px rgba(0, 0, 0, 0.25)",
            "button_text": "#ffffff",
            "code_bg": "#0f172a",
            "code_text": "#e2e8f0",
        }
    else:  # Light theme
        return {
            "bg_gradient_1": "#f8fafc",
            "bg_gradient_2": "#e2e8f0",
            "bg_gradient_3": "#f1f5f9",
            "text_primary": "#000000",      # Pure black
            "text_secondary": "#333333",    # Dark gray
            "card_bg": "rgba(255, 255, 255, 0.85)",
            "card_border": "rgba(203, 213, 225, 0.6)",
            "sidebar_bg": "rgba(255, 255, 255, 0.98)",
            "user_bubble": "#2563eb",
            "user_text": "#ffffff",
            "assistant_bubble": "rgba(255, 255, 255, 0.9)",
            "assistant_text": "#1e293b",
            "accent": "#2563eb",
            "accent_glow": "rgba(37, 99, 235, 0.3)",
            "input_bg": "rgba(255, 255, 255, 0.8)",
            "input_border": "rgba(37, 99, 235, 0.2)",
            "shadow": "0 3px 6px rgba(15, 23, 42, 0.15)",
            "button_text": "#ffffff",
            "code_bg": "#f1f5f9",
            "code_text": "#334155",
        }


# ---------- Global CSS Injection ----------
def inject_css(theme):
    colors = get_theme_colors(theme)
    font_size_map = {"Small": "14px", "Medium": "16px", "Large": "18px"}
    current_font_size = font_size_map.get(st.session_state.settings["font_size"], "16px")

    # Light mode sidebar CSS (BLACK TEXT + Model dropdown WHITE on BLUE)
    light_sidebar_css = """
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #000000 !important;
    }
    /* LIGHT MODE: Model dropdown WHITE TEXT on BLUE BG */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        color: #ffffff !important;
        background: rgba(37, 99, 235, 0.8) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] option {
        color: #000000 !important;
        background: #ffffff !important;
    }
    [data-testid="stSidebar"] [role="combobox"] {
        color: #ffffff !important;
    }
    """ if theme == "Light" else ""

    # Dark mode sidebar CSS
    dark_sidebar_css = """
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }
    """ if theme == "Dark" else ""

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body {{
        font-family: 'Inter', sans-serif;
        font-size: {current_font_size};
        color: {colors['text_primary']};
        background-color: transparent;
    }}

    .stApp {{
        background: linear-gradient(-45deg, {colors['bg_gradient_1']}, {colors['bg_gradient_2']}, {colors['bg_gradient_3']}, {colors['bg_gradient_1']});
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        min-height: 100vh;
        padding-top: 0;
    }}

    @keyframes gradientBG {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    header[data-testid="stHeader"], footer {{
        visibility: hidden;
        height: 0;
        position: fixed;
    }}

    .main .block-container {{
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }}

    [data-testid="stSidebar"] {{
        background: {colors['sidebar_bg']} !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-right: 1px solid {colors['card_border']} !important;
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
        padding-left: 0.6rem;
        padding-right: 0.6rem;
        transition: width 0.3s ease, padding 0.3s ease;
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
    }}

    /* LIGHT MODE SIDEBAR */
    {light_sidebar_css}

    /* DARK MODE SIDEBAR */
    {dark_sidebar_css}

    [data-testid="stSidebar"] > div:first-child {{
        background: transparent;
        width: 100%;
    }}

    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: transparent;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {colors['card_border']};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {colors['text_secondary']};
    }}

    .stButton > button {{
        background: linear-gradient(135deg, {colors['accent']}, #1d4ed8) !important;
        color: {colors['button_text']} !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        padding: 0.25rem 0.6rem !important;
        font-size: 0.84rem !important;
        box-shadow: {colors['shadow']};
        transition: all 0.25s ease !important;
        position: relative;
        overflow: hidden;
        min-height: 32px;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 8px 16px -4px {colors['accent_glow']};
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {{
        background: {colors['input_bg']} !important;
        border: 1px solid {colors['input_border']} !important;
        color: {colors['text_primary']} !important;
        border-radius: 10px !important;
        backdrop-filter: blur(4px);
        padding-top: 0.3rem !important;
        padding-bottom: 0.3rem !important;
    }}

    [data-testid="stChatInput"] > div {{
        background: {colors['input_bg']};
        border: 1px solid {colors['input_border']};
        border-radius: 18px;
        box-shadow: {colors['shadow']};
        backdrop-filter: blur(10px);
        padding: 0.15rem 0.6rem !important;
        transition: all 0.3s ease;
        min-height: 42px;
    }}

    [data-testid="stChatInput"] textarea {{
        min-height: 24px !important;
        max-height: 80px !important;
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
    }}

    [data-testid="stChatInput"] > div:focus-within {{
        border-color: {colors['accent']};
        box-shadow: 0 0 0 2px {colors['accent_glow']};
    }}

    .chat-message {{
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.9rem;
        animation: fadeIn 0.3s ease-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .user-message {{
        justify-content: flex-end;
    }}

    .avatar {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 6px;
        flex-shrink: 0;
        box-shadow: {colors['shadow']};
        font-size: 0.9rem;
    }}

    .message-content {{
        padding: 0.6rem 0.75rem;
        border-radius: 14px;
        max-width: 80%;
        position: relative;
        box-shadow: {colors['shadow']};
        backdrop-filter: blur(6px);
        word-wrap: break-word;
        font-size: 0.92rem;
    }}

    .user-message .message-content {{
        background: {colors['user_bubble']};
        color: {colors['user_text']};
        border-radius: 14px 14px 3px 14px;
    }}

    .assistant-message .message-content {{
        background: {colors['assistant_bubble']};
        color: {colors['assistant_text']};
        border-radius: 14px 14px 14px 3px;
        border: 1px solid {colors['card_border']};
    }}

    .mode-card {{
        background: {colors['card_bg']};
        border: 1px solid {colors['card_border']};
        border-radius: 12px;
        padding: 0.5rem 0.6rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.25s ease;
        backdrop-filter: blur(8px);
        box-shadow: {colors['shadow']};
        height: 100%;
        color: {colors['text_primary']};
        margin-bottom: 0.4rem;
    }}

    .mode-card div:first-child {{
        font-size: 1.4rem;
        margin-bottom: 0.25rem;
    }}

    .mode-card div:nth-child(2) {{
        font-size: 0.9rem;
    }}

    .mode-card div:nth-child(3) {{
        font-size: 0.7rem;
    }}

    .mode-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 14px {colors['accent_glow']};
        border-color: {colors['accent']};
    }}

    .mode-card.active {{
        background: linear-gradient(135deg, {colors['accent']}, #1d4ed8);
        color: white;
        border: none;
    }}

    .file-preview {{
        background: {colors['card_bg']};
        border: 1px solid {colors['card_border']};
        border-radius: 10px;
        padding: 0.6rem 0.75rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        backdrop-filter: blur(8px);
        box-shadow: {colors['shadow']};
        color: {colors['text_primary']};
        font-size: 0.86rem;
    }}

    .welcome-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        position: relative;
        z-index: 10;
    }}

    .welcome-card {{
        background: {colors['card_bg']};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {colors['card_border']};
        border-radius: 20px;
        padding: 2rem 2.2rem;
        width: 100%;
        max-width: 420px;
        box-shadow: 0 22px 40px -14px {colors['accent_glow']};
        animation: slideUp 0.4s ease-out;
        color: {colors['text_primary']};
    }}

    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .logo-container {{
        width: 96px;
        height: 96px;
        background: linear-gradient(135deg, {colors['accent']}, #1d4ed8);
        border-radius: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 46px;
        margin: 0 auto 1.4rem;
        box-shadow: 0 18px 32px {colors['accent_glow']};
        animation: float 6s ease-in-out infinite;
    }}

    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
        100% {{ transform: translateY(0px); }}
    }}

    pre {{
        background: {colors['code_bg']} !important;
        color: {colors['code_text']} !important;
        border-radius: 6px !important;
        padding: 0.7rem !important;
        border: 1px solid {colors['card_border']} !important;
        font-size: 0.86rem !important;
    }}

    .streamlit-expanderHeader {{
        background: {colors['card_bg']} !important;
        color: {colors['text_primary']} !important;
        border: 1px solid {colors['card_border']} !important;
        border-radius: 6px !important;
        padding-top: 0.35rem !important;
        padding-bottom: 0.35rem !important;
        font-size: 0.9rem !important;
    }}

    [data-testid="stSidebarHeader"] > button {{
        background: transparent !important;
        color: {colors['text_primary']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const toggleButton = document.querySelector('[data-testid="stSidebarHeader"] > button');
        if (toggleButton) {
            toggleButton.addEventListener('click', function() {
                setTimeout(() => {
                    const sidebar = document.querySelector('[data-testid="stSidebar"]');
                    if (sidebar) {
                        sidebar.style.display = 'none';
                        sidebar.offsetHeight;
                        sidebar.style.display = '';
                    }
                }, 100);
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)


# Inject CSS immediately
inject_css(st.session_state.settings["theme"])


# ---------- Welcome gate ----------
if not st.session_state.settings["user_name"] or not st.session_state.settings["role"]:
    colors = get_theme_colors(st.session_state.settings["theme"])

    st.markdown(f"""
    <div class="welcome-container">
        <div class="logo-container">💻</div>
        <h1 style="text-align: center; margin-bottom: 0.3rem; font-weight: 700; font-size: 2.1rem; color: {colors['text_primary']};">Code Gen Ai</h1>
        <p style="text-align: center; margin-bottom: 1.2rem; opacity: 0.8; color: {colors['text_secondary']};">Personalized coding workspace</p>
        <h2 style="text-align: center; margin-bottom: 0.9rem; font-weight: 600; font-size: 1.4rem; color: {colors['text_primary']};">Enter your details</h2>
    </div>
    """, unsafe_allow_html=True)

    name = st.text_input(
        "👤 Your name",
        value=st.session_state.settings.get("user_name") or "",
        placeholder="Enter your name here..."
    )

    role = st.radio(
        "🎯 Who are you?",
        ["Student", "Teacher", "Coder", "Employee", "Business"],
        horizontal=True
    )

    if st.button("🚀 Continue", use_container_width=True):
        if name.strip():
            st.session_state.settings["user_name"] = name.strip()
            st.session_state.settings["role"] = role.lower()
            st.session_state["mode"] = get_modes_for_role(role)[0]
            st.rerun()
        else:
            st.error("⚠️ Please enter your name!")

    st.stop()


# ---------- Session State Initialization ----------
if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []

if "active_thread_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.chat_threads.append({
                "id": new_id,
        "title": "New Chat",
        "messages": [],
        "created": datetime.now()
    })
    st.session_state.active_thread_id = new_id

if "ocr_context" not in st.session_state:
    st.session_state.ocr_context = {"text": None, "filename": None}

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None

if "show_uploader" not in st.session_state:
    st.session_state["show_uploader"] = False

if "mode" not in st.session_state:
    st.session_state["mode"] = get_modes_for_role(st.session_state.settings["role"])[0]
if "processing" not in st.session_state:
    st.session_state.processing = False
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Set it in .env file")
    st.stop()

# ---------- Helper Functions ----------
def get_active_thread():
    for thread in st.session_state.chat_threads:
        if thread["id"] == st.session_state.active_thread_id:
            return thread
    return st.session_state.chat_threads[0]


def create_new_chat():
    new_id = str(uuid.uuid4())
    new_thread = {
        "id": new_id,
        "title": "New Chat",
        "messages": [],
        "created": datetime.now()
    }
    st.session_state.chat_threads.insert(0, new_thread)
    st.session_state.active_thread_id = new_id


def delete_thread(thread_id):
    st.session_state.chat_threads = [t for t in st.session_state.chat_threads if t["id"] != thread_id]
    if not st.session_state.chat_threads:
        create_new_chat()
    else:
        st.session_state.active_thread_id = st.session_state.chat_threads[0]["id"]


def rename_thread(thread_id, new_name):
    for thread in st.session_state.chat_threads:
        if thread["id"] == thread_id:
            thread["title"] = new_name
            break


def derive_thread_title(thread):
    existing = thread.get("title") or "New Chat"
    messages = thread.get("messages", [])
    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content"):
            snippet = message["content"].splitlines()[0]
            if len(snippet) > 36:
                snippet = snippet[:33] + "..."
            thread["title"] = snippet or existing
            return thread["title"]
    return existing


def generate_title(text):
    title = " ".join(text.split())
    title = title[0].upper() + title[1:] if title else "New Chat"
    return title[:40] + "..." if len(title) > 40 else title


def call_groq_api(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,
            max_tokens=4000,
            stream=False
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {str(e)}"


def recognize_speech():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            with st.spinner("Listening..."):
                audio = r.listen(source, phrase_time_limit=5)
        return r.recognize_google(audio)
    except Exception:
        return None


def extract_text_from_image(file_obj):
    try:
        file_obj.seek(0)
        img = Image.open(file_obj)
        text = pytesseract.image_to_string(img).strip()
        cleaned = " ".join(text.split())
        st.success(f"✅ OCR: {len(cleaned)} chars")
        return cleaned[:4000]
    except Exception as e:
        st.error(f"❌ OCR: {e}")
        return ""


# ---------- Sidebar ----------
with st.sidebar:
    colors = get_theme_colors(st.session_state.settings["theme"])

    st.markdown(f"""
    <div style='text-align: left; padding: 0 0 0.6rem 0; border-bottom: 1px solid {colors['card_border']}; margin-bottom: 0.8rem;'>
        <div style='display:flex; align-items:center; gap:0.6rem;'>
            <div style='width:32px;height:32px;border-radius:10px;
                        background:linear-gradient(135deg,{colors['accent']},#1d4ed8);
                        display:flex;align-items:center;justify-content:center;font-size:18px; box-shadow: {colors['shadow']};'>💻</div>
            <div>
                <div style='color:{colors['text_primary']};font-weight:600;font-size:0.98rem;'>Code Gen Ai</div>
                <div style='color:{colors['text_secondary']};font-size:0.75rem;'>Coding copilot</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    theme_choice = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.settings["theme"] == "Dark" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    if theme_choice != st.session_state.settings["theme"]:
        st.session_state.settings["theme"] = theme_choice
        st.rerun()  # CRITICAL: Re-injects CSS with new theme

    st.markdown("##### Models")
    base_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]
    options = base_models + ["Mock Mode (Demo)"]
    st.session_state.settings["model"] = st.selectbox("Active model", options, label_visibility="collapsed")

    st.markdown("##### Preferences")
    st.session_state.settings["particles"] = st.toggle(
        "Background pattern",
        value=st.session_state.settings["particles"],
        label_visibility="visible"
    )
    st.session_state.settings["font_size"] = st.select_slider(
        "Font size",
        options=["Small", "Medium", "Large"],
        value=st.session_state.settings["font_size"],
        label_visibility="visible"
    )

    st.markdown("##### Chats")
    col_new, col_clear = st.columns([0.7, 0.3])
    with col_new:
        if st.button("＋ New", use_container_width=True):
            create_new_chat()
            st.rerun()
    with col_clear:
        if st.button("🗑️ All"):
            st.session_state.chat_threads = []
            create_new_chat()
            st.rerun()

    for thread in st.session_state.chat_threads[-10:]:
        if st.button(f"💬 {derive_thread_title(thread)[:22]}", key=thread["id"], use_container_width=True):
            st.session_state.active_thread_id = thread["id"]
            st.rerun()


# ---------- Main Content Area ----------
active_thread = get_active_thread()
messages = active_thread["messages"]
user_name = st.session_state.settings.get("user_name", "Coder")
role = st.session_state.settings.get("role", "student")
current_modes = get_modes_for_role(role)
colors = get_theme_colors(st.session_state.settings["theme"])


# Header (compact)
st.markdown(f"""
<div style='display:flex; align-items:center; gap:14px; margin: 0 0 0.7rem 0; padding: 0.7rem 0.9rem; background: {colors['card_bg']}; border-radius: 14px; border: 1px solid {colors['card_border']}; backdrop-filter: blur(8px); box-shadow: {colors['shadow']};'>
<div style='width:48px; height:48px; border-radius:16px; 
            background:radial-gradient(circle at 30% 10%, {colors['accent']}, #1d4ed8);
            display:flex; align-items:center; justify-content:center; color:white; font-size:26px;
            box-shadow: {colors['shadow']};'>
 💻
</div>
<div>
        <div style='font-weight:700; color:{colors['text_primary']}; font-size:1.25rem;'>Code Gen Ai</div>
    <div style='font-size:0.8rem; opacity:0.85; color: {colors['text_secondary']};'>
     Hey {user_name}, {role} mode is active. Pick what you want to do.
    </div>
</div>
</div>
""", unsafe_allow_html=True)


# Mode Selector (compact, 4 small cards)
icons_for_mode = {
    "Debug code": "🐞",
    "Solve problem": "✅",
    "Explain code": "📖",
    "Practise code": "🧑‍💻",
    "Learn new technology": "🚀",
}

mode_cols = st.columns(4)
for i, mode_name in enumerate(current_modes[:4]):
    col = mode_cols[i]
    icon = icons_for_mode.get(mode_name, "✨")
    is_active = st.session_state["mode"] == mode_name

    with col:
        st.markdown(f"""
        <div class="mode-card {'active' if is_active else ''}" onclick="document.getElementById('mode_{i}').click()">
            <div>{icon}</div>
            <div>{mode_name}</div>
            <div>{'Active' if is_active else 'Tap to switch'}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(mode_name, key=f"mode_{i}", use_container_width=True):
            st.session_state["mode"] = mode_name
            st.rerun()


# Chat Messages
chat_container = st.container()
with chat_container:
    if not messages:
        st.markdown(
            f"<div style='color:{colors['text_secondary']}; margin-top:0.8rem; text-align: center; font-size:0.9rem;'>Type a question or paste code below. Mode changes how Code Gen Ai responds.</div>",
            unsafe_allow_html=True,
        )
    else:
        for msg in messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-content">
                        <div style="font-size: 0.7rem; opacity: 0.7; margin-bottom: 0.25rem; text-align: right;">
                            You • {msg.get("timestamp", "now")}
                        </div>
                        <div style="white-space: pre-wrap;">{msg['content']}</div>
                    </div>
                    <div class="avatar" style="background: linear-gradient(135deg, {colors['accent']}, #1d4ed8); color: white;">
                        👤
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif msg["role"] == "assistant":
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="avatar" style="background: linear-gradient(135deg, {colors['accent']}, #1d4ed8); color: white;">
                        💻
                    </div>
                    <div class="message-content">
                        <div style="font-size: 0.7rem; opacity: 0.7; margin-bottom: 0.25rem;">
                            Code Gen Ai • {msg.get("timestamp", "now")}
                        </div>
                        <div style="white-space: pre-wrap;">{msg['content']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# AI Response
if st.session_state.get("processing", False) and st.session_state.get("last_prompt", ""):
    if "generating_response" not in st.session_state:
        st.session_state.generating_response = True

        model = st.session_state.settings.get("model", "llama-3.1-8b-instant")
        if model == "Mock Mode (Demo)":
            answer = "**Mock Mode:** This is a demo response."
        else:
            with st.spinner("Code Gen Ai is thinking..."):
                answer = call_groq_api(st.session_state.last_prompt, model)

        active_thread["messages"].append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        st.session_state.processing = False
        st.session_state.last_prompt = ""
        if "generating_response" in st.session_state:
            del st.session_state.generating_response
        st.rerun()


# File Preview
if st.session_state.get("last_file_name"):
    is_image = st.session_state['last_file_name'].lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))

    st.markdown(f"""
    <div class="file-preview">
        <div style="display: flex; align-items: center; gap: 0.6rem;">
            {'<div style="font-size: 22px;">🖼️</div>' if is_image else '<div style="font-size: 22px;">📎</div>'}
            <div>
                <div style="font-weight: 600;">{st.session_state['last_file_name']}</div>
                <div style="font-size: 0.75rem; opacity: 0.7;">{len(st.session_state.ocr_context.get("text", ""))} characters extracted</div>
            </div>
        </div>
        <button onclick="document.getElementById('cancel_file').click()" style="background: transparent; border: none; color: {colors['text_secondary']}; cursor: pointer; font-size: 1rem;">✕</button>
        <button id="cancel_file" style="display: none;"></button>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✕", key="cancel_file_preview_unique"):
        st.session_state.ocr_context = {"text": None, "filename": None, "image_bytes": None}
        st.session_state.last_file_name = None
        st.rerun()

    ocr_text = st.session_state.ocr_context.get("text", "")
    if ocr_text:
        with st.expander(f"🔍 OCR Extracted Text ({len(ocr_text)} chars)"):
            st.text_area("Content", ocr_text[:300], height=120, key="ocr_debug")
    else:
        st.error("❌ NO TEXT EXTRACTED - Tesseract issue!")


# Bottom Bar - compact
with st.container():
    col_input, col_icons = st.columns([0.88, 0.12])

    with col_input:
        user_input = st.chat_input("Ask about code, errors, or upload screenshots...")

    with col_icons:
        c1, c2 = st.columns(2)
        with c1:
            attach_clicked = st.button("📎", help="Attach file", use_container_width=True)
        with c2:
            mic_clicked = st.button("🎤", help="Speak", use_container_width=True)


# Attach File Logic
if attach_clicked:
    st.session_state["show_uploader"] = True

if st.session_state.get("show_uploader", False):
    uploaded_quick = st.file_uploader(
        "Upload",
        type=["py", "txt", "png", "jpg", "jpeg", "pdf"],
        key="quick_upl",
        label_visibility="collapsed"
    )
    if uploaded_quick is not None:
        fname = uploaded_quick.name
        try:
            is_image = fname.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))

            if is_image:
                text = extract_text_from_image(uploaded_quick)
            elif fname.lower().endswith(".pdf") and OCR_AVAILABLE:
                text = ""
                pages = convert_from_bytes(uploaded_quick.read())
                for p in pages:
                    text += pytesseract.image_to_string(p) + "\n"
                text = " ".join(text.split())
            else:
                text = uploaded_quick.read().decode("utf-8", errors="ignore")
                text = " ".join(text.split())

            if text:
                st.session_state.ocr_context = {
                    "text": text,
                    "filename": fname,
                    "image_bytes": None
                }
                st.session_state.last_file_name = fname
                st.success(f"Attached {fname}. Type your question.")

                if st.session_state.ocr_context.get("text"):
                    mode = st.session_state.get("mode", "Debug code")
                    ocr_text = st.session_state.ocr_context["text"]
                    filename = st.session_state.ocr_context["filename"]

                    auto_prompt = (
                        f"{BASE_MODE_PROMPTS[mode]}\n\n"
                        f"**Screenshot/File:** {filename} ({len(ocr_text)} chars extracted):\n"
                        f"{ocr_text}\n\n"
                        f"**TASK:** Analyze this screenshot/code."
                    )

                    st.session_state.last_prompt = auto_prompt
                    st.session_state.processing = True

        except Exception as e:
            st.error(f"Could not read {fname}: {e}")

        st.session_state["show_uploader"] = False
        st.rerun()


# Mic Logic
if mic_clicked:
    spoken = recognize_speech()
    if spoken:
        mode = st.session_state.get("mode")
        system_prefix = BASE_MODE_PROMPTS.get(mode, "")
        final_prompt = f"{system_prefix}\n\nUser query:\n{spoken}"

        active_thread["messages"].append({
            "role": "user",
            "content": spoken,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        if len(active_thread["messages"]) <= 2:
            rename_thread(active_thread["id"], generate_title(spoken))

        st.session_state.last_prompt = final_prompt
        st.session_state.processing = True
        st.rerun()


# Text Input Logic
if user_input:
    mode = st.session_state.get("mode")
    system_prefix = BASE_MODE_PROMPTS.get(mode, "")

    final_prompt = f"{system_prefix}\n\nUser query:\n{user_input}"

    if st.session_state.ocr_context.get("text"):
        ocr_text = st.session_state.ocr_context["text"]
        filename = st.session_state.ocr_context["filename"]
        final_prompt = (
            f"{system_prefix}\n\n"
            f"**Screenshot/File:** {filename}\n"
            f"**OCR Extracted Code/UI:**\n"
            f"{ocr_text}\n\n"
            f"**User Question:** {user_input}"
        )
        st.session_state.ocr_context = {"text": None, "filename": None}
        st.session_state.last_file_name = None

    with st.expander(f"🔍 AI Prompt ({len(final_prompt)} chars)"):
        st.code(final_prompt, language="text")

    active_thread["messages"].append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().strftime("%H:%M")
    })

    if len(active_thread["messages"]) <= 2:
        rename_thread(active_thread["id"], generate_title(user_input))

    st.session_state.last_prompt = final_prompt
    st.session_state.processing = True
    st.rerun()

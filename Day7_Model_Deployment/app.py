
import streamlit as st
from transformers import pipeline

# ─────────────────────────────────────────────
# 🎯 App Setup
st.set_page_config(page_title="Smart LLM Assistant", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background-color: #111827;
        color: #f9fafb;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput>div>div>input {
        background-color: #1f2937;
        color: white;
    }
    .stButton>button {
        background-color: #ef4444;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stMarkdown h1 {
        font-size: 2.5rem;
        color: #facc15;
    }
    .response {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        color: #f1f5f9;
        font-size: 1.1rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("# 🤖 Smart LLM Assistant")
st.markdown("##### *Powered by flan-t5-large + few-shot prompting*")

# ─────────────────────────────────────────────
# 🧠 Load the model (cached)
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=0  # GPU
    )

generator = load_model()

# ─────────────────────────────────────────────
# 🧩 Prompt Constructor
def build_prompt(user_q: str):
    demo = (
        "Q: What is the Python programming language?\n"
        "A: Python is a high-level, interpreted programming language known for its readability, "
        "versatility, and vast ecosystem of libraries. It’s used in web development, data science, automation, and more.\n\n"
    )
    instr = ""
    q_lower = user_q.lower()
    if any(k in q_lower for k in ["write", "generate", "example", "function"]):
        instr = "Write Python code for the following task:\n"
    elif any(k in q_lower for k in ["step-by-step", "how to solve", "calculate"]):
        instr = "Solve this problem step by step:\n"
    else:
        instr = "Explain clearly and concisely:\n"
    return demo + instr + user_q + "\nA:"

# ─────────────────────────────────────────────
# 🎯 App Interface
st.markdown("#### 💬 Ask your question below:")

q = st.text_input("Enter a question, task, or prompt:")

if st.button("🚀 Generate Response"):
    if not q.strip():
        st.warning("⚠️ Please enter a valid prompt.")
    else:
        prompt = build_prompt(q)
        with st.spinner("🧠 Thinking..."):
            out = generator(
                prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                num_return_sequences=1
            )[0]["generated_text"]

        answer = out.split("A:")[-1].strip()

        st.markdown("### 🎯 Answer:")
        st.markdown(f'<div class="response">{answer}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 📎 Footer
st.markdown("---")
st.caption("⚙️ Model: google/flan-t5-large | Prompting: demo + instruction → natural answers")

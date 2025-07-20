import streamlit as st
from transformers import pipeline

# 1) Load a stronger model (fits in ~4 GB VRAM)
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=0  # GPU
    )

generator = load_model()

# 2) Few-shot + template builder
def build_prompt(user_q: str):
    # A tiny demonstration showing Q→A style
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
    # Combine demo + instruction + actual question
    return demo + instr + user_q + "\nA:"

# 3) Streamlit UI
st.set_page_config(page_title="Smart LLM Assistant", layout="centered")
st.title("🤖 Smart LLM Assistant (flan-t5-large + few-shot)")

q = st.text_input("💬 Your question or task:")
if st.button("🚀 Go"):
    if not q.strip():
        st.warning("Please enter a question or task.")
    else:
        prompt = build_prompt(q)
        with st.spinner("Generating…"):
            out = generator(
                prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                num_return_sequences=1
            )[0]["generated_text"]
        # Trim off the “A:” label if echoed
        answer = out.split("A:")[-1].strip()
        st.markdown("### 🤖 Answer:")
        st.write(answer)

# Footer
st.markdown("---")
st.caption("Model: google/flan-t5-large | Format demo + instruction → robust answers")

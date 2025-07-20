# 🧠 LLM Sprint – Days 7 to 10: Deployment, Evaluation & Optimization

This phase of the sprint focuses on taking your trained Large Language Model (LLM) from experimentation to production readiness. It covers deploying the model as a web app, evaluating performance, tuning hyperparameters, and monitoring deployed models.

---

## 📅 Day 7 – Model Deployment with Streamlit

- Implemented a **Streamlit-based UI** to interact with the `google/flan-t5-large` model.
- Used `transformers.pipeline` for `text2text-generation`.
- Integrated few-shot prompt formatting for dynamic tasks (code generation, explanation, etc).
- Deployed locally and explored hosting options:
  - Streamlit Community Cloud
  - Hugging Face Spaces
  - Docker + Cloud VMs (AWS/Heroku)

✅ **Key file:** `app.py`  
✅ **Concepts:** UI design, caching models, GPU usage, user input handling

---

## 📅 Day 8 – Model Evaluation

- Evaluated model performance using:
  - Manual review of generated text
  - Response quality across various prompt types
- Discussed potential metrics for generative tasks:
  - BLEU, ROUGE, METEOR, GPTScore
- Observed how temperature, top-p, and repetition penalties affect output diversity.

✅ **Focus:** Accuracy vs creativity trade-off  
✅ **Tools used:** Hugging Face `pipeline`, manual QA review

---

## 📅 Day 9 – Hyperparameter Tuning

- Tuned decoding parameters like:
  - `temperature`, `top_p`, `repetition_penalty`, `max_new_tokens`
- Created variations for exploratory runs
- Documented the best combinations for:
  - Coding tasks
  - Explanation-style prompts
  - Step-by-step problem solving

✅ **Outcome:** Achieved balance between coherence and diversity  
✅ **Bonus:** Discussed integration of Optuna/WandB for future auto-tuning

---

## 📅 Day 10 – CI/CD + Monitoring

- Designed a lightweight CI/CD plan using:
  - GitHub → Streamlit Cloud auto-deploy on push
  - Hugging Face Spaces deploy on commit
- Considered ML observability:
  - Logging input/output pairs
  - Storing feedback for future fine-tuning
- Setup `.gitignore`, `requirements.txt`, and basic folder structure for team collaboration.

✅ **Deliverable:** Public app URL + reusable deployment setup  
✅ **Future Work:** Add analytics/logging (e.g., Streamlit session_state, Sentry, Prometheus)



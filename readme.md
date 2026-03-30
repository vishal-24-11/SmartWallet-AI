# 🤖 AI-Driven Personal Finance Categorizer Engine

## 📌 Project Overview
An end-to-end Data Science pipeline that transforms unstructured, real-world bank statements into actionable financial insights. This project utilizes Natural Language Processing (NLP) to map messy transaction narrations to structured categories, visualizes spending habits, and deploys a linear programming mathematical optimizer to recommend prescriptive budget cuts.

## 🚀 Key Features
* **NLP Categorization Engine:** Uses a `TfidfVectorizer` and `RandomForestClassifier` to read and categorize Indian banking UPI formats and local vendor strings.
* **Entity Resolution:** Dynamically intercepts and re-categorizes specific entities like 'Self Transfers' and 'Rent' based on user inputs.
* **Exploratory Data Analysis (EDA):** A dynamic dashboard featuring multi-period filtering, Matplotlib/Seaborn visualizations, and tabular aggregations.
* **Prescriptive "What-If" Optimizer:** Integrates the `pulp` linear programming library to perform goal-seeking budget optimization, minimizing the "pain" of cutting expenses while mathematically adhering to essential living constraints.

## 🛠️ Technology Stack
* **Machine Learning & NLP:** `scikit-learn` (Random Forest, TF-IDF)
* **Optimization Analytics:** `pulp` (Linear Programming)
* **Data Engineering:** `pandas`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Web Deployment:** `gradio` (Deployed via Hugging Face Spaces)

## 📂 Repository Structure
* `Finance_Categorizer_Main_Pipeline.ipynb`: The core Google Colab notebook containing the entire training, testing, and EDA pipeline.
* `app.py`: The Gradio web application source code.
* `requirements.txt`: The required Python library dependencies.
* `financial_transaction_train1.csv`: The primary training dataset.

---
*Developed as a Data Science Capstone Project.*

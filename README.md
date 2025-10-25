
# AI Symptom Checker

A simple AI-powered symptom checker built with **Streamlit** and **scikit-learn**.  
Enter your symptoms to get potential disease predictions based on a trained decision tree model.

> ⚠️ **Disclaimer:** This tool is for informational purposes only and is **not a substitute for professional medical advice**. Always consult a healthcare provider.

---

## Features

- Select symptoms from a pre-defined list or add your own.
- Display selected symptoms in real-time.
- Predict possible disease using a trained Decision Tree model.
- Lightweight, fast, and responsive UI.

---

## Project Structure

```
└── 📁ai-symptom-checker
    └── 📁.streamlit
        ├── config.toml
    └── 📁src
        └── 📁data
            ├── testing_data.csv
            ├── training_data.csv
        └── 📁models
            ├── model.py
        └── 📁utils
            ├── helpers.py
        ├── app.py
        ├── assets
    ├── README.md
    └── requirements.txt
```


---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ai-symptom-checker.git
cd ai-symptom-checker

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

Install dependencies:

pip install -r requirements.txt

Usage

Run the Streamlit app:

streamlit run src/app.py


The app will open in your browser. You can:

Select symptoms from the dropdown list.

Add custom symptoms via text input.

Click "Get Prediction" to see possible diseases.

How it Works

Loads a CSV dataset (training_data.csv) containing symptoms and corresponding diseases.

Trains a simple Decision Tree classifier.

Encodes user-selected symptoms into a 0/1 vector.

Uses the trained model to predict a disease.

Displays the result in a clean, responsive UI.

Dependencies

See requirements.txt
.
---

### **`requirements.txt`**

```txt
streamlit>=1.25.0
pandas>=2.1.0
scikit-learn>=1.3.0
numpy>=1.26.0
=======

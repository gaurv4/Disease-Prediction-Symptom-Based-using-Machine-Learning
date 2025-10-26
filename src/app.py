import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.model import train_top_models, load_top_models, ensemble_predict
from utils.helpers import encode_symptoms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---- Symptom Categories ----
symptom_categories = {
    "Stomach / Digestion": {
        'back_pain': "Back hurts",
        'constipation': "Can't poop well",
        'abdominal_pain': "Stomach hurts",
        'diarrhoea': "Loose poop",
        'mild_fever': "Slight fever",
        'yellow_urine': "Urine looks yellow",
        'yellowing_of_eyes': "Eyes look yellow",
        'swelling_of_stomach': "Stomach bloated",
        'malaise': "Feeling weak",
        'bloody_stool': "Poop has blood",
    },
    "Breathing / Chest": {
        'phlegm': "Mucus in throat",
        'throat_irritation': "Sore throat",
        'runny_nose': "Nose runs",
        'congestion': "Nose blocked",
        'chest_pain': "Chest hurts",
        'coughing_blood': "Cough with blood",
        'mucoid_sputum': "Mucus in cough",
    },
    "Body / Pain / Movement": {
        'neck_pain': "Neck hurts",
        'dizziness': "Feel dizzy",
        'cramps': "Muscle cramps",
        'weakness_in_limbs': "Arms/legs weak",
        'swollen_legs': "Legs swollen",
        'stiff_neck': "Neck stiff",
        'movement_stiffness': "Hard to move",
        'painful_walking': "Pain while walking",
        'knee_pain': "Knee hurts",
    },
    "Mind / Senses / Mood": {
        'blurred_and_distorted_vision': "Vision blurry",
        'depression': "Feeling sad",
        'irritability': "Easily annoyed",
        'lack_of_concentration': "Hard to focus",
        'slurred_speech': "Talk unclear",
    },
    "Skin / Nails": {
        'red_spots_over_body': "Red spots on skin",
        'pus_filled_pimples': "Pimples with pus",
        'blackheads': "Black spots on skin",
        'skin_peeling': "Skin peeling",
        'blister': "Fluid-filled bump",
    },
    "Other / Misc": {
        'excessive_hunger': "Always hungry",
        'family_history': "Family sickness history",
        'polyuria': "Pee often",
        'palpitations': "Heart skipping beats",
    }
}

# ---- Disease Descriptions ----
disease_descriptions = {
    "Fungal infection": "A skin infection caused by fungi, leading to itching, redness, or peeling.",
    "Allergy": "A reaction by your immune system to something harmless, like pollen or dust.",
    "GERD": "A digestive issue where stomach acid flows back into the esophagus, causing heartburn.",
    "Chronic cholestasis": "A liver-related condition that causes yellowing of eyes and itching.",
    "Drug Reaction": "An unwanted effect caused by medication, like rashes or swelling.",
    "Peptic ulcer disease": "Painful sores in the stomach lining, often causing stomach pain or nausea.",
    "Diabetes": "A condition where blood sugar levels are too high.",
    "Gastroenteritis": "An infection that causes vomiting, diarrhea, and stomach cramps.",
    "Hypertension": "Also known as high blood pressure — may cause headaches or dizziness.",
    "Migraine": "A type of headache that can cause severe throbbing pain and sensitivity to light or sound.",
    "Bronchial Asthma": "A condition causing wheezing, shortness of breath, and coughing.",
    "Jaundice": "Yellowing of skin and eyes due to liver issues.",
    "Malaria": "A mosquito-borne infection causing fever, chills, and body aches.",
    "Pneumonia": "A lung infection that causes coughing, fever, and difficulty breathing.",
    "Tuberculosis": "A bacterial infection that mainly affects the lungs, causing cough and fever.",
    "Typhoid": "A bacterial infection from contaminated food/water that causes fever and weakness.",
    "Dengue": "A mosquito-borne viral illness causing fever, body pain, and rashes.",
    "Covid-19": "A viral infection causing fever, cough, and loss of taste or smell.",
    "Common Cold": "A mild viral infection of the nose and throat — runny nose, sore throat, and cough.",
    "Heart Disease": "A range of heart-related problems causing chest pain or shortness of breath.",
    "Arthritis": "Joint inflammation causing stiffness and pain.",
    "Paralysis (brain hemorrhage)": "Loss of muscle control due to brain injury or stroke.",
    "Hypothyroidism": "When the thyroid gland is underactive, causing fatigue and weight gain.",
    "Hyperthyroidism": "An overactive thyroid causing weight loss, anxiety, and rapid heartbeat.",
    "Varicose veins": "Swollen, twisted veins usually visible on the legs.",
    "Cervical spondylosis": "Neck pain from wear and tear of the spinal discs.",
    "Chicken pox": "A viral infection causing itchy red spots and fever.",
    "Psoriasis": "A skin condition causing red, scaly patches.",
    "Impetigo": "A contagious skin infection that causes red sores, often on the face."
}


# ---- Streamlit Page Setup ----
st.set_page_config(page_title="AI Disease Prediction", layout="wide")
st.title("🩺 AI-Powered Disease Prediction Dashboard")

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("src/data/training_data.csv")
    X = df.drop(columns=['prognosis'])
    y = df['prognosis']
    all_symptoms = X.columns.tolist()
    all_diseases = sorted(y.unique().tolist())
    return df, X, y, all_symptoms, all_diseases

df, X, y, all_symptoms, all_diseases = load_data()

# ---- Train or Load Models ----
@st.cache_resource
def load_models():
    top_models = train_top_models(X, y)
    return load_top_models(), top_models

models, top_models = load_models()

# ---- Sidebar ----
st.sidebar.title("⚙️ App Controls")
st.sidebar.info("Use this sidebar to explore model metrics, dataset stats, or make predictions.")

option = st.sidebar.radio(
    "Choose a Section:",
    ["🧠 Disease Prediction", "📊 Model Performance", "📈 Data Visualization", "ℹ️ About"]
)

# ---- SECTION 1: DISEASE PREDICTION ----
if option == "🧠 Disease Prediction":
    st.header("Symptom-based Disease Prediction")

    # --- Select Symptoms ---
    # --- Select Symptoms by Category ---
    st.markdown("### 🩺 Select Your Symptoms")

    selected_symptoms = []

# Option 1: Browse by Category

    with st.expander("Select Symptoms by Category"):
        for category, symptoms in symptom_categories.items():
            st.markdown(f"**{category}**")
            cols = st.columns(2)
            half = len(symptoms) // 2 + 1
        # Split into two columns for better layout
            for i, (symptom_key, symptom_label) in enumerate(symptoms.items()):
                col = cols[0] if i < half else cols[1]
                if col.checkbox(symptom_label, key=symptom_key):
                    selected_symptoms.append(symptom_key)


    if st.button("🔮 Predict Disease"):
            if not selected_symptoms:
                st.error("Please select at least one symptom.")
            else:
                # Encode symptoms
                input_vector = np.array(encode_symptoms(selected_symptoms, all_symptoms)).reshape(1, -1)

        # Predict using ensemble
            probs, pred_disease = ensemble_predict(models, input_vector, all_diseases)

        # ---- Display Prediction Result ----
            desc = disease_descriptions.get(pred_disease, "Description not available for this disease.")

            st.markdown(f"""
                <div style="padding:20px; border-radius:15px; background-color:#4CAF50; text-align:center;">
                <h2 style="color:white;">Predicted Disease</h2>
                <h1 style="color:white;">{pred_disease}</h1>
                <h3 style="color:white;">Confidence: {max(probs)*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            st.info(f"🩺 **About {pred_disease}:** {desc}")


        # ---- Show Other Possible Diseases ----
            st.subheader("🩺 Other Possible Diseases")
            prob_df = pd.DataFrame({'Disease': all_diseases, 'Probability': probs})
            prob_df = prob_df.sort_values('Probability', ascending=False)
            st.dataframe(prob_df.head(10))
            st.bar_chart(prob_df.head(10).set_index('Disease'))

        # ---- Medical Disclaimer ----
            st.warning("⚠️ This AI tool is not a medical diagnosis. Please consult a licensed doctor for professional advice.")

# ---- SECTION 2: MODEL PERFORMANCE ----
elif option == "📊 Model Performance":
            st.header("📊 Model Evaluation and Comparison")

            y_true = y
            y_pred = ensemble_predict(models, X.values, all_diseases, return_all=True)

            metrics = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision (Macro)': precision_score(y_true, y_pred, average='macro'),
                'Recall (Macro)': recall_score(y_true, y_pred, average='macro'),
                'F1 Score (Macro)': f1_score(y_true, y_pred, average='macro')
            }

            st.subheader("Overall Ensemble Model Performance")
            st.write(pd.DataFrame(metrics, index=["Score"]).T)

            # ---- Model Comparison ----
            st.subheader("Individual Model Performance (Top 3)")
            perf_df = pd.DataFrame(top_models, columns=["Model", "Accuracy"]).sort_values("Accuracy", ascending=False)
            st.dataframe(perf_df)
            st.bar_chart(perf_df.set_index("Model"))

            # ---- Confusion Matrix ----
            st.subheader("Confusion Matrix (Ensemble)")
            cm = confusion_matrix(y_true, y_pred, labels=all_diseases)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=False, fmt='d', xticklabels=all_diseases, yticklabels=all_diseases, cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(plt)

# ---- SECTION 3: DATA VISUALIZATION ----
elif option == "📈 Data Visualization":
            st.header("📈 Explore Dataset Insights")

        # Symptom Frequency
            st.subheader("🩺 Top 20 Most Common Symptoms")
            symptom_counts = X.sum().sort_values(ascending=False).head(20)
            st.bar_chart(symptom_counts)

        # Disease Distribution
            st.subheader("📊 Disease Distribution in Dataset")
            disease_counts = y.value_counts()
            st.bar_chart(disease_counts)

        # Correlation heatmap
            st.subheader("📉 Symptom Correlation Heatmap")
            corr = X.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, cmap="coolwarm", cbar=True)
            st.pyplot(plt)

        # Feature importance (if available)
            try:
                rf = [m for m in models if hasattr(m, "feature_importances_")][0]
                importances = pd.Series(rf.feature_importances_, index=all_symptoms).sort_values(ascending=False)
                st.subheader("🔥 Top 15 Most Important Symptoms (Random Forest)")
                st.bar_chart(importances.head(15))
            except:
                st.info("Feature importance unavailable — model does not support it.")

# ---- SECTION 4: ABOUT ----
elif option == "ℹ️ About":
            st.header("About the AI Symptom Checker")
            st.markdown("""
                This AI system predicts possible diseases based on user-selected symptoms using **machine learning models**:
                - 🧠 **XGBoost**
                - 🌳 **Random Forest**
                - 💡 **LightGBM**
                - ⚙️ **Logistic Regression**

                It uses an **ensemble learning approach** — combining multiple model predictions for better accuracy and stability.

                ### 💡 Key Features
                - Predicts likely disease and confidence score  
                - Shows top 10 probable diseases  
                - Displays model performance metrics (accuracy, F1, etc.)  
                - Provides interactive data visualizations  
                - Explains symptom importance  
                - Clean UI built with Streamlit  

                **Note:** This app is for educational purposes and is *not a substitute for professional medical advice.*
                """)
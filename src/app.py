import streamlit as st
import pandas as pd
from models.model import train_decision_tree
from utils.helpers import encode_symptoms, predict_disease
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("src/data/training_data.csv")
    X = df.drop(columns=['prognosis'])
    y = df['prognosis']
    all_symptoms = X.columns.tolist()
    all_diseases = y.unique().tolist()
    return X, y, all_symptoms, all_diseases

X, y, all_symptoms, all_diseases = load_data()

# ---- Train Model ----
@st.cache_data
def load_model():
    return train_decision_tree(X, y)

model = load_model()

# ---- Symptom Categories with Easy Descriptions ----
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
        'acute_liver_failure': "Serious liver problem",
        'fluid_overload': "Water retention",
        'swelled_lymph_nodes': "Lumps in neck/armpits",
        'pain_during_bowel_movements': "Pain while pooping",
        'pain_in_anal_region': "Pain near anus",
        'bloody_stool': "Poop has blood",
        'irritation_in_anus': "Itchy/burning anus",
        'belly_pain': "Tummy hurts",
        'stomach_bleeding': "Bleeding in stomach",
        'distention_of_abdomen': "Big swollen tummy",
        'history_of_alcohol_consumption': "Drinked alcohol often"
    },
    "Breathing / Chest": {
        'phlegm': "Mucus in throat",
        'throat_irritation': "Sore throat",
        'redness_of_eyes': "Red eyes",
        'sinus_pressure': "Face feels tight",
        'runny_nose': "Nose runs",
        'congestion': "Nose blocked",
        'chest_pain': "Chest hurts",
        'coughing_blood': "Cough with blood",
        'foul_smell_of_urine': "Bad smell when peeing",
        'continuous_feel_of_urine': "Always feel need to pee",
        'bladder_discomfort': "Pain while peeing",
        'mucoid_sputum': "Mucus in cough",
        'rusty_sputum': "Cough with rusty mucus",
        'blood_in_sputum': "Cough with blood"
    },
    "Body / Pain / Movement": {
        'neck_pain': "Neck hurts",
        'dizziness': "Feel dizzy",
        'cramps': "Muscle cramps",
        'muscle_pain': "Muscles hurt",
        'weakness_in_limbs': "Arms/legs weak",
        'fast_heart_rate': "Heart beats fast",
        'obesity': "Overweight",
        'swollen_legs': "Legs swollen",
        'swollen_blood_vessels': "Veins swollen",
        'puffy_face_and_eyes': "Face swollen",
        'enlarged_thyroid': "Neck gland swollen",
        'brittle_nails': "Nails break easily",
        'swollen_extremeties': "Hands/feet swollen",
        'muscle_weakness': "Weak muscles",
        'stiff_neck': "Neck stiff",
        'swelling_joints': "Joints swollen",
        'movement_stiffness': "Hard to move",
        'spinning_movements': "Feel spinning",
        'loss_of_balance': "Can't balance well",
        'unsteadiness': "Feeling unstable",
        'weakness_of_one_body_side': "One side weak",
        'painful_walking': "Pain while walking",
        'knee_pain': "Knee hurts",
        'hip_joint_pain': "Hip hurts"
    },
    "Mind / Senses / Mood": {
        'blurred_and_distorted_vision': "Vision blurry",
        'loss_of_smell': "Can't smell",
        'depression': "Feeling sad",
        'irritability': "Easily annoyed",
        'lack_of_concentration': "Hard to focus",
        'visual_disturbances': "Vision problems",
        'altered_sensorium': "Confused",
        'slurred_speech': "Talk unclear",
        'coma': "Unconscious"
    },
    "Skin / Nails": {
        'red_spots_over_body': "Red spots on skin",
        'dischromic _patches': "Skin color changes",
        'watering_from_eyes': "Eyes water",
        'pus_filled_pimples': "Pimples with pus",
        'blackheads': "Black spots on skin",
        'scurring': "Flaky skin",
        'skin_peeling': "Skin peeling",
        'silver_like_dusting': "Skin shiny patches",
        'small_dents_in_nails': "Small dents on nails",
        'inflammatory_nails': "Nail inflammation",
        'blister': "Fluid-filled bump",
        'red_sore_around_nose': "Red sore near nose",
        'yellow_crust_ooze': "Yellow crust on skin"
    },
    "Other / Misc": {
        'excessive_hunger': "Always hungry",
        'extra_marital_contacts': "Risky sexual contact",
        'drying_and_tingling_lips': "Dry/tingly lips",
        'polyuria': "Pee often",
        'increased_appetite': "Eat more",
        'family_history': "Family sickness history",
        'receiving_blood_transfusion': "Got blood transfusion",
        'receiving_unsterile_injections': "Unsafe injections",
        'palpitations': "Heart skipping beats",
        'toxic_look_(typhos)': "Looks very sick"
    }
}

# ---- Streamlit App ----
st.set_page_config(page_title="Disease Prediction", layout="wide")
st.title("🩺 Easy Disease Prediction")
st.write("Click what you feel. Simple words, no medical terms.")

# ---- Symptom Input ----
selected_symptoms = []
for category, symptoms in symptom_categories.items():
    with st.expander(category, expanded=False):
        cols = st.columns(3)
        for i, (symptom, desc) in enumerate(symptoms.items()):
            col = cols[i % 3]
            if col.checkbox(f"{desc}", key=symptom):
                selected_symptoms.append(symptom)

# ---- Show Selected Symptoms ----
if selected_symptoms:
    st.write("**You feel:**", ", ".join([symptom.replace("_", " ") for symptom in selected_symptoms]))
else:
    st.info("Select at least one feeling to check.")

# ---- Prediction & Result ----
if st.button("Check Possible Disease"):
    if not selected_symptoms:
        st.error("Please select at least one feeling!")
    else:
        import numpy as np

        # Encode symptoms and reshape to 2D
        symptom_vector = encode_symptoms(selected_symptoms, all_symptoms)
        symptom_vector_2d = np.array(symptom_vector).reshape(1, -1)

        # Predict
        probs = model.predict_proba(symptom_vector_2d)[0]
        prediction = all_diseases[np.argmax(probs)]

        # ---- Display Prediction Result ----
        st.markdown(f"""
        <div style="padding:20px; border-radius:15px; background-color:#3dc4a1; text-align:center;">
            <h2 style="color:#fff;">Prediction Result</h2>
            <div style="font-size:36px; font-weight:bold; color:#fff;">{max(probs)*100:.1f}%</div>
            <h3 style="margin-top:10px; color:#fff;">{prediction}</h3>
            <p style="color:#fff;">Based on the symptoms you provided, there is a statistical likelihood that you may have <b>{prediction}</b>.</p>
            <div style="margin-top:15px; background-color:#fff3cd; padding:10px; border-radius:10px; color:#856404;">
                ⚠ This is not a medical diagnosis. Please consult a qualified healthcare professional.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Top Predicted Diseases ----
        st.subheader("🔮 Top Predicted Diseases for You")
        prob_df = pd.DataFrame({'Disease': all_diseases, 'Probability': probs}).sort_values('Probability', ascending=False)
        st.dataframe(prob_df.head(10))
        st.bar_chart(prob_df.head(10).set_index('Disease'))


        # ---- Model Performance Metrics ----
        st.subheader("📊 Model Performance on Dataset")
        y_pred = model.predict(X)
        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision (Macro)': precision_score(y, y_pred, average='macro'),
            'Recall (Macro)': recall_score(y, y_pred, average='macro'),
            'F1 Score (Macro)': f1_score(y, y_pred, average='macro')
        }
        st.write(metrics)

        # ---- Confusion Matrix ----
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred, labels=all_diseases)
        plt.figure(figsize=(12,10))
        sns.heatmap(cm, annot=False, fmt='d', xticklabels=all_diseases, yticklabels=all_diseases, cmap="Blues")
        plt.xlabel("Predicted Disease")
        plt.ylabel("Actual Disease")
        st.pyplot(plt)

        # ---- Symptom Distribution ----
        st.subheader("🩺 Symptom Occurrence in Dataset")
        symptom_counts = X.sum().sort_values(ascending=False)
        st.bar_chart(symptom_counts[:20])

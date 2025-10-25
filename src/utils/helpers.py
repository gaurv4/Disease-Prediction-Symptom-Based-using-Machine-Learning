import numpy as np

def encode_symptoms(selected_symptoms, all_symptoms):
    """
    Convert selected symptoms into a binary vector for the model.
    """
    vector = np.zeros(len(all_symptoms))
    for i, symptom in enumerate(all_symptoms):
        if symptom in selected_symptoms:
            vector[i] = 1
    return vector.reshape(1, -1)

def predict_disease(model, symptom_vector, all_diseases):
    """
    Predict disease using the trained model.
    """
    pred_index = model.predict(symptom_vector)[0]
    return pred_index


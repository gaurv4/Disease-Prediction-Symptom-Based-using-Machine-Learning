import numpy as np

def encode_symptoms(selected_symptoms, all_symptoms):
    """
    Encode the list of selected symptoms into a binary feature vector.

    Parameters:
        selected_symptoms (list): Symptoms chosen by the user.
        all_symptoms (list): All possible symptoms (from the dataset columns).

    Returns:
        list: Binary vector (1 = symptom present, 0 = absent)
    """
    encoded = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
    return encoded


def preprocess_input(symptoms, all_symptoms):
    """
    (Optional helper) Convert to numpy 2D array for model input.
    """
    return np.array([encode_symptoms(symptoms, all_symptoms)])

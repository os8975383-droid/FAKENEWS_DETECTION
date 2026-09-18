import os
import sys
import joblib

from preprocessing import clean_text


MODEL_PATH = "fake_news_pipeline.pkl"


def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Model not found.\n"
            "Run train.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_news(text):

    model = load_model()

    cleaned_text = clean_text(text)

    prediction = model.predict([cleaned_text])[0]

    # LinearSVC does not have predict_proba.
    # decision_function is used to calculate
    # a normalized confidence-like score.
    decision = model.decision_function([cleaned_text])[0]

    if prediction == 1:

        label = "REAL"

    else:

        label = "FAKE"

    # Convert decision value into a probability-like score.
    # This is NOT a calibrated probability.
    import math

    score = 1 / (1 + math.exp(-float(decision)))

    if prediction == 0:
        score = 1 - score

    return label, score


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            'Usage:\n'
            'python src/predict.py "Your news article here"'
        )

        sys.exit()

    text = " ".join(sys.argv[1:])

    label, confidence = predict_news(text)

    print("\nPrediction:", label)
    print(f"Confidence-like score: {confidence * 100:.2f}%")
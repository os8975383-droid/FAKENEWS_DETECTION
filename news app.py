import os
import math
import joblib
import streamlit as st

from preprocessing import clean_text


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


MODEL_PATH = "models/fake_news_pipeline.pkl"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📰 Fake News Detection System")

st.write(
    """
    Enter a news headline or article below and the machine-learning
    model will classify the text as **REAL** or **FAKE**.
    """
)


st.info(
    "⚠️ This is an ML classifier, not a fact-checking engine. "
    "A prediction should not be treated as proof that a news story "
    "is factually true or false."
)


# --------------------------------------------------
# MODEL CHECK
# --------------------------------------------------

if model is None:

    st.error(
        "Trained model not found. "
        "Run `python src/train.py` first."
    )

    st.stop()


# --------------------------------------------------
# INPUT
# --------------------------------------------------

news_text = st.text_area(
    "Paste news article / headline",
    height=250,
    placeholder="Enter the news text here..."
)


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

if st.button(
    "🔍 Detect Fake News",
    use_container_width=True
):

    if not news_text.strip():

        st.warning(
            "Please enter some news text."
        )

    else:

        cleaned_text = clean_text(news_text)

        prediction = model.predict(
            [cleaned_text]
        )[0]
        st.write("Raw prediction:", prediction)
        st.write("Model classes:", model.classes_)

        decision = model.decision_function(
            [cleaned_text]
        )[0]
        

        # Convert decision value to confidence-like score
        score = 1 / (
            1 + math.exp(-float(decision))
        )

        if prediction == 1:

            label = "REAL"

        else:

            label = "FAKE"
            score = 1 - score


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.divider()

        st.subheader("Prediction")

        if label == "REAL":

            st.success(
                f"### ✅ {label}"
            )

        else:

            st.error(
                f"### 🚨 {label}"
            )


        st.metric(
            "Confidence-like score",
            f"{score * 100:.2f}%"
        )


        st.progress(
            min(max(score, 0.0), 1.0)
        )


        st.caption(
            "The score represents the model's classification "
            "confidence-like measure, not independent verification "
            "of the claim."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Fake News Detection | Machine Learning + NLP + TF-IDF"
)

# ============================================================
# FAKE NEWS DETECTION USING MACHINE LEARNING
# WELFake Dataset - Kaggle
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import re
import glob
import json
import html
import unicodedata

import kagglehub
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. PROJECT SETTINGS
# ============================================================

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 3. DOWNLOAD DATASET FROM KAGGLE
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOADING WELFAKE DATASET FROM KAGGLE")
print("=" * 70)

path = kagglehub.dataset_download(
    "saurabhshahane/fake-news-classification"
)

print("\nDataset downloaded successfully!")
print("Path to dataset files:")
print(path)


# ============================================================
# 4. FIND CSV FILE AUTOMATICALLY
# ============================================================

print("\n" + "=" * 70)
print("SEARCHING FOR CSV FILE")
print("=" * 70)

csv_files = glob.glob(
    os.path.join(path, "**", "*.csv"),
    recursive=True
)

if not csv_files:

    raise FileNotFoundError(
        "\nNo CSV file found inside the downloaded Kaggle dataset.\n"
        f"Dataset path: {path}"
    )

print("\nCSV files found:")

for i, file in enumerate(csv_files, start=1):
    print(f"{i}. {file}")


# ------------------------------------------------------------
# Select CSV
# ------------------------------------------------------------

DATA_PATH = csv_files[0]

print("\nUsing dataset:")
print(DATA_PATH)


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns available:")
print(df.columns.tolist())


# ============================================================
# 6. NORMALIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

print("\nNormalized columns:")
print(df.columns.tolist())


# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

if "label" not in df.columns:

    raise ValueError(
        "\n'label' column was not found in the dataset."
        f"\nAvailable columns: {df.columns.tolist()}"
    )


# ============================================================
# 8. CREATE NEWS TEXT
# ============================================================

print("\n" + "=" * 70)
print("PREPARING TEXT DATA")
print("=" * 70)


# WELFake contains title + text.
# We combine both because title can contain useful information.

if "title" in df.columns and "text" in df.columns:

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df["news_text"] = (
        df["title"].astype(str)
        + " "
        + df["text"].astype(str)
    )

elif "text" in df.columns:

    df["news_text"] = df["text"].fillna("").astype(str)

elif "title" in df.columns:

    df["news_text"] = df["title"].fillna("").astype(str)

else:

    raise ValueError(
        "\nNeither 'title' nor 'text' column was found."
        f"\nAvailable columns: {df.columns.tolist()}"
    )


# ============================================================
# 9. CLEAN LABELS
# ============================================================

print("\nChecking labels...")

print(df["label"].value_counts(dropna=False))


# WELFake:
# 0 = Fake
# 1 = Real

def convert_label(value):

    value = str(value).strip().lower()

    if value in ["0", "fake", "false", "f"]:

        return 0

    elif value in ["1", "real", "true", "r"]:

        return 1

    else:

        return np.nan


df["label"] = df["label"].apply(convert_label)


# Remove invalid labels

df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)


# ============================================================
# 10. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    if pd.isna(text):

        return ""

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Convert lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Keep letters and spaces
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("\nCleaning news text...")

df["clean_text"] = df["news_text"].apply(clean_text)


# ============================================================
# 11. REMOVE EMPTY TEXT
# ============================================================

before = len(df)

df = df[df["clean_text"].str.len() > 0]

after = len(df)

print(f"\nRemoved empty rows: {before - after}")


# ============================================================
# 12. REMOVE DUPLICATE NEWS
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["clean_text"]
)

after = len(df)

print(f"Removed duplicate articles: {before - after}")


# ============================================================
# 13. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print("\nFinal dataset shape:")
print(df.shape)

print("\nLabel distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)


print("\nLabel percentages:")

print(
    df["label"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# ============================================================
# 14. BASIC EDA - LABEL DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="label"
)

plt.title("Fake vs Real News Distribution")

plt.xlabel("Label")
plt.ylabel("Number of Articles")

plt.xticks(
    [0, 1],
    ["Fake", "Real"]
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "label_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 15. TEXT LENGTH ANALYSIS
# ============================================================

df["text_length"] = (
    df["clean_text"]
    .str.len()
)


plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="text_length",
    hue="label",
    bins=50,
    kde=True
)

plt.title("News Text Length Distribution")

plt.xlabel("Number of Characters")

plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "text_length_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 16. PREPARE X AND Y
# ============================================================

X = df["clean_text"]

y = df["label"]


print("\nX shape:", X.shape)

print("y shape:", y.shape)


# ============================================================
# 17. TRAIN TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ============================================================
# 18. TF-IDF SETTINGS
# ============================================================

def create_vectorizer():

    return TfidfVectorizer(

        lowercase=False,

        stop_words="english",

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95,

        sublinear_tf=True,

        max_features=100000
    )


# ============================================================
# 19. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "Multinomial Naive Bayes":
        MultinomialNB(),

    "Linear SVM":
        LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "SGD Classifier":
        SGDClassifier(
            loss="modified_huber",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
}


# ============================================================
# 20. TRAIN MODELS
# ============================================================

results = []

trained_models = {}


print("\n" + "=" * 70)
print("MODEL TRAINING STARTED")
print("=" * 70)


for model_name, model in models.items():

    print("\n" + "-" * 70)

    print(f"Training: {model_name}")

    print("-" * 70)


    # Create a fresh TF-IDF vectorizer
    vectorizer = create_vectorizer()


    # Create pipeline
    pipeline = Pipeline([

        (
            "tfidf",
            vectorizer
        ),

        (
            "model",
            model
        )
    ])


    # Train
    pipeline.fit(
        X_train,
        y_train
    )


    # Prediction
    y_pred = pipeline.predict(
        X_test
    )


    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # Save results
    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1
    })


    # Save trained model in memory
    trained_models[model_name] = pipeline


    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print("\nResults:")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )


    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Fake",
                "Real"
            ],
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    plt.figure(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Fake",
            "Real"
        ],
        yticklabels=[
            "Fake",
            "Real"
        ]
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()


    # Safe filename
    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )


    plt.savefig(
        os.path.join(
            MODEL_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 21. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 22. SAVE MODEL RESULTS
# ============================================================

results_df.to_csv(

    os.path.join(
        MODEL_DIR,
        "model_results.csv"
    ),

    index=False
)


# ============================================================
# 23. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_f1 = results_df.iloc[0]["F1 Score"]

best_pipeline = trained_models[
    best_model_name
]


print("\n" + "=" * 70)

print("BEST MODEL")

print("=" * 70)

print(
    f"\nBest Model: {best_model_name}"
)

print(
    f"Best F1 Score: {best_f1:.4f}"
)


# ============================================================
# 24. SAVE BEST MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fake_news_pipeline.pkl"
)


joblib.dump(
    best_pipeline,
    MODEL_PATH
)


print("\nBest model saved at:")

print(MODEL_PATH)


# ============================================================
# 25. SAVE METADATA
# ============================================================

metadata = {

    "dataset": "WELFake - Kaggle",

    "dataset_source":
        "saurabhshahane/fake-news-classification",

    "label_mapping": {

        "0": "Fake",

        "1": "Real"
    },

    "best_model":
        best_model_name,

    "best_f1_score":
        float(best_f1),

    "training_samples":
        int(len(X_train)),

    "testing_samples":
        int(len(X_test)),

    "total_samples":
        int(len(df)),

    "tfidf": {

        "ngram_range": [
            1,
            2
        ],

        "max_features": 100000,

        "min_df": 2,

        "max_df": 0.95,

        "stop_words": "english",

        "sublinear_tf": True
    }
}


with open(

    os.path.join(
        MODEL_DIR,
        "metadata.json"
    ),

    "w",

    encoding="utf-8"

) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# 26. SAVE CLEAN DATASET
# ============================================================

clean_dataset_path = os.path.join(
    MODEL_DIR,
    "cleaned_news_data.csv"
)


# Save only useful columns
df[
    [
        "clean_text",
        "label"
    ]
].to_csv(

    clean_dataset_path,

    index=False
)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("TRAINING COMPLETED SUCCESSFULLY!")

print("=" * 70)


print("\nProject files created:")

print(
    "1. models/fake_news_pipeline.pkl"
)

print(
    "2. models/model_results.csv"
)

print(
    "3. models/metadata.json"
)

print(
    "4. models/label_distribution.png"
)

print(
    "5. models/text_length_distribution.png"
)

print(
    "6. models/*_confusion_matrix.png"
)

print(
    "7. models/cleaned_news_data.csv"
)


print("\nBest Model:")

print(
    best_model_name
)


print("\nBest F1 Score:")

print(
    f"{best_f1:.4f}"
)


print("\n" + "=" * 70)

print("IMPORTANT:")

print(
    "This model is a text classification model."
)

print(
    "It does NOT verify facts from the internet."
)

print(
    "A high confidence/prediction score does not prove"
)

print(
    "that a news article is objectively true or false."
)

print("=" * 70)
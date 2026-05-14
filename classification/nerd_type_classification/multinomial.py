import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# -------------------------
# SETTINGS
# -------------------------

file_path = "nerd_dataset.csv"

chunk_size = 10_000

numeric_features = [
    "curiosity",
    "organization",
    "chaos",
    "focus_level",
    "research_depth",
    "social_energy",
    "aesthetic_obsession",
    "tool_usage"
]

categorical_features = [
    "favorite_subject",
    "learning_style",
    "favorite_media"
]

target_col = "nerd_type"

overall_start = time.time()


# -------------------------
# TIME FORMATTER
# -------------------------

def format_time(seconds):

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    return f"{minutes}m {remaining_seconds:.2f}s"


# -------------------------
# LOAD DATASET
# -------------------------

print("\nLoading dataset...")

file_size = os.path.getsize(file_path)

chunks = []

bytes_loaded = 0

for chunk in pd.read_csv(file_path, chunksize=chunk_size):

    chunks.append(chunk)

    bytes_loaded += chunk.memory_usage(deep=True).sum()

    percent = min((bytes_loaded / file_size) * 100, 100)

    print(f"{percent:.1f}% done loading dataset...")

df = pd.concat(chunks, ignore_index=True)

print("\n100% done loading dataset.")

print(f"Rows loaded: {len(df):,}")


# -------------------------
# SPLIT DATA
# -------------------------

print("\nSplitting train/test data...")

X = df.drop(columns=["nerd_id", "nerd_type"], errors="ignore")

y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training rows: {len(X_train):,}")

print(f"Testing rows: {len(X_test):,}")


# -------------------------
# PREPROCESSING
# -------------------------

print("\nBuilding preprocessing pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

print("\nPreprocessing training data...")

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("Preprocessing complete.")


# -------------------------
# TRAIN MODEL
# -------------------------

print("\nTraining model in chunks... 😭")

model = SGDClassifier(
    loss="log_loss",
    random_state=42
)

classes = y_train.unique()

training_start = time.time()

train_total = X_train_processed.shape[0]

for start in range(0, train_total, chunk_size):

    end = min(start + chunk_size, train_total)

    X_batch = X_train_processed[start:end]

    y_batch = y_train.iloc[start:end]

    if start == 0:
        model.partial_fit(
            X_batch,
            y_batch,
            classes=classes
        )
    else:
        model.partial_fit(
            X_batch,
            y_batch
        )

    percent = (end / train_total) * 100

    elapsed = time.time() - training_start

    print(
        f"{percent:.1f}% done training... | "
        f"Rows: {end:,}/{train_total:,} | "
        f"Elapsed: {format_time(elapsed)}"
    )

training_end = time.time()

print(
    f"\nTraining complete in "
    f"{format_time(training_end - training_start)}"
)


# -------------------------
# PREDICTIONS
# -------------------------

print("\nGenerating predictions...")

prediction_start = time.time()

predictions = []

test_total = X_test_processed.shape[0]

for start in range(0, test_total, chunk_size):

    end = min(start + chunk_size, test_total)

    X_batch = X_test_processed[start:end]

    batch_predictions = model.predict(X_batch)

    predictions.extend(batch_predictions)

    percent = (end / test_total) * 100

    elapsed = time.time() - prediction_start

    print(
        f"{percent:.1f}% done predicting... | "
        f"Rows: {end:,}/{test_total:,} | "
        f"Elapsed: {format_time(elapsed)}"
    )

prediction_end = time.time()

print(
    f"\nPredictions complete in "
    f"{format_time(prediction_end - prediction_start)}"
)


# -------------------------
# EVALUATION
# -------------------------

print("\nEvaluating model...")

print("\nAccuracy:")

print(accuracy_score(y_test, predictions))

print("\nClassification Report:")

print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, predictions))


# -------------------------
# SAMPLE PREDICTION
# -------------------------

print("\nTesting sample person...")

sample_person = pd.DataFrame([{
    "curiosity": 9,
    "organization": 8,
    "chaos": 2,
    "focus_level": 9,
    "research_depth": 9,
    "social_energy": 2,
    "aesthetic_obsession": 4,
    "tool_usage": 7,
    "favorite_subject": "statistics",
    "learning_style": "solo",
    "favorite_media": "papers"
}])

sample_processed = preprocessor.transform(sample_person)

prediction = model.predict(sample_processed)[0]

probabilities = model.predict_proba(sample_processed)[0]

print("\nPredicted Nerd Type:")

print(prediction)

print("\nProbability Breakdown:")

for nerd_class, prob in zip(model.classes_, probabilities):

    print(f"{nerd_class}: {prob:.2%}")


# -------------------------
# TOTAL RUNTIME
# -------------------------

overall_end = time.time()

print("\nML pipeline complete 😭")

print(
    f"Total Runtime: "
    f"{format_time(overall_end - overall_start)}"
)

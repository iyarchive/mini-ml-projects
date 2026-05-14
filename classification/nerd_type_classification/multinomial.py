import pandas as pd

# splits dataset into training and testing sets
from sklearn.model_selection import train_test_split

# converts categorical strings into numbers + scales numeric features
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# applies different preprocessing methods to different columns
from sklearn.compose import ColumnTransformer

# chains preprocessing + model into one workflow
from sklearn.pipeline import Pipeline

# multinomial logistic regression classification model
from sklearn.linear_model import LogisticRegression

# evaluates model performance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("nerd_dataset.csv")

X = df.drop(["nerd_id", "nerd_type"], axis=1)
y = df["nerd_type"]

numeric_features = ["curiosity", "organization", "chaos"]
categorical_features = ["favorite_subject", "learning_style", "favorite_media"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


sample_person = pd.DataFrame([{
    "curiosity": 9,
    "organization": 8,
    "chaos": 3,
    "favorite_subject": "statistics",
    "learning_style": "solo",
    "favorite_media": "papers"
}])

prediction = model.predict(sample_person)[0]
probabilities = model.predict_proba(sample_person)[0]

print("\nSample Prediction:")
print(prediction)

print("\nProbability Breakdown:")
for nerd_class, prob in zip(model.classes_, probabilities):
    print(f"{nerd_class}: {prob:.2%}")

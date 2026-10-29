
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv(
    r"C:\Users\Bhavi M Patel\Desktop\resume project\Resume_Cleaned.csv"
)


df = df[["Resume_str", "Category"]].dropna()
df["Resume_str"] = df["Resume_str"].astype(str)
df["Category"] = df["Category"].astype(str)
df = df[df["Resume_str"].str.strip() != ""]


X = df["Resume_str"]
y = df["Category"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    )),
    ("classifier", LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced"
    ))
])


print("Training model...")
model.fit(X_train, y_train)
print("Training completed!")




y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))


print("\n--- RESUME JOB ROLE PREDICTION ---")
print("Paste your resume text below.")
print("Type END on a new line when finished:\n")

lines = []

while True:
    line = input()
    if line.strip().upper() == "END":
        break
    lines.append(line)

resume_text = "\n".join(lines)


if resume_text.strip():
    predicted_role = model.predict([resume_text])[0]
    print("\nPredicted Job Role:", predicted_role)
else:
    print("No resume text entered.")

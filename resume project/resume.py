
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv(
    r"C:\Users\Bhavi M Patel\Desktop\resume project\Resume_Cleaned.csv"
)

# 2. Clean dataset
df = df[["Resume_str", "Category"]].dropna()
df["Resume_str"] = df["Resume_str"].astype(str)
df["Category"] = df["Category"].astype(str)
df = df[df["Resume_str"].str.strip() != ""]

# 3. Input and output
X = df["Resume_str"]
y = df["Category"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Logistic Regression model
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

# 6. Train model
print("Training model...")
model.fit(X_train, y_train)
print("Training completed!")

# 7. Accuracy
print("\nTraining Accuracy:",
      round(model.score(X_train, y_train) * 100, 2), "%")

print("Testing Accuracy:",
      round(model.score(X_test, y_test) * 100, 2), "%")

# 8. Classification report
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# 9. Enter resume text
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

# 10. Predict job role
if resume_text.strip():
    predicted_role = model.predict([resume_text])[0]
    print("\nPredicted Job Role:", predicted_role)
else:
    print("No resume text entered.")

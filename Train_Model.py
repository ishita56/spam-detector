import os
import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Current working directory:", os.getcwd())

# Load dataset
df = pd.read_csv(
    r"C:\Users\ishit\OneDrive\Desktop\spam detector\archive (2)\spam.csv",
    encoding='latin-1'
)

# Keep required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['message'] = df['message'].apply(clean_text)

# Label encoding
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label_num'],
    test_size=0.2,
    random_state=42,
    stratify=df['label_num']
)

# Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# =========================
# =========================
base_path = os.path.dirname(os.path.abspath(__file__))

pickle.dump(model, open(os.path.join(base_path, "model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(base_path, "vectorizer.pkl"), "wb"))

print("Model and vectorizer saved successfully!")

# =========================
# OPTIONAL TEST LOOP
# =========================
while True:
    msg = input("Enter a message (or 'exit'): ")

    if msg.lower() == 'exit':
        break

    cleaned_msg = clean_text(msg)
    msg_vec = vectorizer.transform([cleaned_msg])

    prediction = model.predict(msg_vec)[0]

    print("Spam" if prediction == 1 else "Ham")
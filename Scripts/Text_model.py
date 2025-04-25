import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

nltk.download('punkt')

# Load dataset
df = pd.read_excel("concat_transcripts_for_text_model.xlsx")
df_clean = df[['Transcript', 'Anxiety_severity']].dropna()
df_clean['clean_text'] = df_clean['Transcript'].str.lower().str.replace(r"<.*?>", "", regex=True)\
                                   .str.replace(r"http\S+", "", regex=True)\
                                   .str.replace(r"[^\w\s]", "", regex=True)

# ================================
# MODEL A: Logistic Regression
# ================================
class CustomAnxietyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, texts):
        tokenizer = TweetTokenizer()
        def count(words, tokens): return sum(1 for w in tokens if w in words)
        f1, f2, f3, f4, f5 = (
            {"i", "me", "my", "mine", "myself"},
            {"worried", "afraid", "nervous", "anxious", "scared"},
            {"maybe", "guess", "not", "unsure", "possibly"},
            {"always", "never", "completely", "nothing", "everything"},
            {"heart", "breath", "sweat", "shaky", "tired", "sleep"},
        )
        feats = []
        for text in texts:
            tokens = tokenizer.tokenize(text)
            feats.append([
                count(f1, tokens), count(f2, tokens),
                count(f3, tokens), count(f4, tokens), count(f5, tokens)
            ])
        return np.array(feats)

# Prepare data
X_text = df_clean['clean_text'].values
y = df_clean['Anxiety_severity'].astype(int).values
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, stratify=y, random_state=42)

# Logistic Regression Features
tfidf = TfidfVectorizer(min_df=3, max_features=1000)
combined_features = FeatureUnion([
    ("tfidf", tfidf),
    ("custom", CustomAnxietyFeatures())
])
X_train_A = combined_features.fit_transform(X_train_text)
X_test_A = combined_features.transform(X_test_text)

# Train Logistic Regression
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model_lr.fit(X_train_A, y_train)
probs_A = model_lr.predict_proba(X_test_A)[:, 1]
y_pred_A = (probs_A > 0.5).astype(int)
acc_A = accuracy_score(y_test, y_pred_A)

# ================================
# MODEL B: LSTM
# ================================
tokenizer_dl = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer_dl.fit_on_texts(X_train_text)
X_train_seq = tokenizer_dl.texts_to_sequences(X_train_text)
X_test_seq = tokenizer_dl.texts_to_sequences(X_test_text)
X_train_B = pad_sequences(X_train_seq, maxlen=200)
X_test_B = pad_sequences(X_test_seq, maxlen=200)

# Compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Build improved LSTM model
model_lstm = Sequential([
    Embedding(10000, 64, input_length=200),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# Train LSTM
history = model_lstm.fit(
    X_train_B, y_train,
    validation_data=(X_test_B, y_test),
    epochs=20, batch_size=16, verbose=1,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# Evaluate LSTM
probs_B = model_lstm.predict(X_test_B).flatten()
y_pred_B = (probs_B > 0.5).astype(int)
acc_B = accuracy_score(y_test, y_pred_B)

# ================================
# ENSEMBLE
# ================================
avg_probs = (0.7 * probs_A + 0.3 * probs_B)
y_ensemble = (avg_probs > 0.5).astype(int)
acc_ensemble = accuracy_score(y_test, y_ensemble)

# ================================
# EVALUATION
# ================================
print("\n--- Individual Model Accuracy ---")
print(f"TF-IDF + Logistic Regression Accuracy: {acc_A:.3f}")
print(f"LSTM Accuracy: {acc_B:.3f}")
print(f"Ensemble Accuracy: {acc_ensemble:.3f}")

print("\n--- Ensemble Classification Report ---")
print(classification_report(y_test, y_ensemble))
print("Ensemble Confusion Matrix:\n", confusion_matrix(y_test, y_ensemble))

# ================================
# PLOTS
# ================================
plt.figure(figsize=(12, 5))

# LSTM Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title("LSTM Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Accuracy Comparison
plt.subplot(1, 2, 2)
models = ['TF-IDF + LR', 'LSTM', 'Ensemble']
accuracies = [acc_A, acc_B, acc_ensemble]
plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'orange'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(axis='y')

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ===============================
# 1. Load and validate dataset
# ===============================
file_path = r"D:\AI Projects\Hackthon Chatbot\chatbot_training_dataset_20000.xlsx"
df = pd.read_excel(file_path)

print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Null values:\n{df.isnull().sum()}")

# Clean the data
df = df.dropna()  # Remove rows with missing values
df = df.drop_duplicates()  # Remove duplicate rows

# Check for questions that are identical to answers (common issue)
df = df[df['question'] != df['answer']]

questions = df["question"].astype(str).values
answers = df["answer"].astype(str).values

print(f"After cleaning: {len(questions)} samples")
print(f"Unique questions: {len(set(questions))}")
print(f"Unique answers: {len(set(answers))}")

# ===============================
# 2. Split data before processing to avoid leakage
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    questions, answers, test_size=0.1, random_state=42, stratify=answers
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# ===============================
# 3. Encode labels (answers)
# ===============================
label_encoder = LabelEncoder()
label_encoder.fit(answers)  # Fit on all answers

y_train_encoded = label_encoder.transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

y_train_cat = to_categorical(y_train_encoded)
y_val_cat = to_categorical(y_val_encoded)

print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"y_train_cat shape: {y_train_cat.shape}")

# ===============================
# 4. Tokenize questions
# ===============================
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)  # Only fit on training data

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_padded = pad_sequences(X_train_seq, maxlen=30, padding="post")
X_val_padded = pad_sequences(X_val_seq, maxlen=30, padding="post")

print(f"Vocabulary size: {len(tokenizer.word_index)}")

# ===============================
# 5. Build model (simplified)
# ===============================
model = Sequential()
model.add(Embedding(input_dim=20000, output_dim=64, input_length=30))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(y_train_cat.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# ===============================
# 6. Train model
# ===============================
checkpoint = ModelCheckpoint("chatbot_model.h5", monitor="val_accuracy", save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Starting training...")
history = model.fit(
    X_train_padded,
    y_train_cat,
    epochs=100,  # Reduced from 200
    batch_size=64,
    validation_data=(X_val_padded, y_val_cat),
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# Save tokenizer and label encoder
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Training completed. Model saved as chatbot_model.h5")

# ===============================
# 7. Accuracy & Loss Chart
# ===============================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_accuracy_loss.png")
plt.show()


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# ==================== CONFIG ====================
# Update these paths to where your files are located
MODEL_DIR = r"D:\AI Projects\Hackthon Chatbot\Model"
MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

def load_model_files():
    """Load model files with error handling"""
    try:
        # Use local copies of paths to avoid modifying globals
        model_path = MODEL_PATH
        tokenizer_path = TOKENIZER_PATH
        label_encoder_path = LABEL_ENCODER_PATH
        # Check if files exist
        if not all(os.path.exists(path) for path in [model_path, tokenizer_path, label_encoder_path]):
            print("❌ Model files not found. Searching...")
            
            # Search for files in the project directory
            for root, dirs, files in os.walk(MODEL_DIR):
                for file in files:
                    if file == "chatbot_model.h5":
                        model_path = os.path.join(root, file)
                    elif file == "tokenizer.pkl":
                        tokenizer_path = os.path.join(root, file)
                    elif file == "label_encoder.pkl":
                        label_encoder_path = os.path.join(root, file)
            
            print(f"Model path: {model_path}")
            print(f"Tokenizer path: {tokenizer_path}")
            print(f"Label encoder path: {label_encoder_path}")
            
            # Check again after search
            if not all(os.path.exists(path) for path in [model_path, tokenizer_path, label_encoder_path]):
                raise FileNotFoundError("Model files not found. Please run training first.")
        
        # Load files
        print("Loading model...")
        model = load_model(model_path)
        
        print("Loading tokenizer...")
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        
        print("Loading label encoder...")
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        
        print("✅ All files loaded successfully!")
        return model, tokenizer, label_encoder
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None

def predict_answer(question, model, tokenizer, label_encoder):
    """Predict answer for a question"""
    try:
        # Preprocess the input question
        sequence = tokenizer.texts_to_sequences([question])
        padded = pad_sequences(sequence, maxlen=30, padding="post")
        
        # Make prediction
        prediction = model.predict(padded, verbose=0)
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction, axis=1)
        answer = label_encoder.inverse_transform(predicted_class)[0]
        
        return answer, confidence
    except Exception as e:
        return f"Error: {e}", 0

# Main execution
if __name__ == "__main__":
    # Load model files
    model, tokenizer, label_encoder = load_model_files()
    
    if model is not None:
        # Test your model
        test_questions = [
            "Hello",
            "Hi there", 
            "Good morning",
            "What can you do?",
            "How are you?",
            "What is your name?"
        ]

        print("\n🧪 Testing the model:\n")
        for question in test_questions:
            answer, confidence = predict_answer(question, model, tokenizer, label_encoder)
            print(f"Q: {question}")
            print(f"A: {answer} (Confidence: {confidence:.2%})")
            print("---")
    else:
        print("Please run your training code first to generate the model files.")
        print("Then make sure the files are in the same directory as this script.")
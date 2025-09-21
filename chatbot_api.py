from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from googletrans import Translator
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Resolve paths relative to this file so the API can run from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")

class HybridChatbot:
    def __init__(self):
        # Load your trained model (ONCE during startup)
        print("Loading AI model...")
        self.model = load_model(os.path.join(MODEL_DIR, "chatbot_model.h5"))
        
        # Load tokenizer and label encoder
        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
            self.tokenizer = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)
        
        # Initialize translator
        self.translator = Translator()
        
        # Initialize Gemini Pro with API key from environment variable
        # Expect an environment variable named GEMINI_API_KEY (loaded from .env if present)
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            # Prefer a current model; fall back to gemini-pro if needed
            try:
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception:
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            self.gemini_available = True
        else:
            print("Warning: GEMINI_API_KEY not found. Gemini fallback disabled.")
            self.gemini_available = False
    
    def should_use_gemini(self, english_question: str) -> bool:
        """Heuristic: route open-domain factual queries directly to Gemini.
        This helps avoid confident-but-wrong local intent classification on facts.
        """
        try:
            q = english_question.lower().strip()
            # Common factual patterns
            factual_triggers = [
                "who is", "who was", "who are", "founder", "co-founder", "when was", "where is",
                "what is", "what are", "how many", "how much", "define ", "meaning of ",
                "capital of", "population of", "invented", "discovered", "age of", "born",
            ]
            return any(trigger in q for trigger in factual_triggers)
        except Exception:
            return False
    
    def translate_to_english(self, text):
        """Translate any language to English"""
        try:
            detected = self.translator.detect(text)
            if detected.lang != 'en':
                translated = self.translator.translate(text, src=detected.lang, dest='en')
                return translated.text, detected.lang
            return text, 'en'
        except:
            return text, 'en'
    
    def translate_to_language(self, text, target_lang):
        """Translate English to target language"""
        try:
            if target_lang != 'en':
                translated = self.translator.translate(text, src='en', dest=target_lang)
                return translated.text
            return text
        except:
            return text
    
    def local_model_predict(self, question):
        """Check if local model can answer the question"""
        try:
            sequence = self.tokenizer.texts_to_sequences([question])
            padded = pad_sequences(sequence, maxlen=30, padding="post")
            
            prediction = self.model.predict(padded, verbose=0)
            confidence = np.max(prediction)
            
            if confidence > 0.9:  # Stricter 90% confidence threshold to reduce wrong local answers
                predicted_class = np.argmax(prediction, axis=1)
                answer = self.label_encoder.inverse_transform(predicted_class)[0]
                return answer, confidence
            return None, confidence
        except:
            return None, 0
    
    def ask_gemini(self, question):
        """Use Gemini Pro API for unanswered questions"""
        if not self.gemini_available:
            return "I'm sorry, I cannot answer that question right now."
        
        try:
            prompt = f"Answer this question concisely in 1-2 sentences: {question}"
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Log the error server-side for debugging
            print(f"[Gemini Error] {e}")
            return "I apologize, but I'm unable to answer that right now. Please verify the GEMINI_API_KEY and network access."

# Initialize chatbot ONCE when server starts
chatbot = HybridChatbot()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Step 1: Translate to English
        english_question, original_lang = chatbot.translate_to_english(user_message)
        
        # Step 2: Strict local-first routing; fallback to Gemini only if local model can't answer
        local_answer, confidence = chatbot.local_model_predict(english_question)
        
        if local_answer:
            response_source = "local_model"
            english_response = local_answer
        else:
            response_source = "gemini_pro"
            english_response = chatbot.ask_gemini(english_question)
        
        # Step 3: Translate response back to user's language
        final_response = chatbot.translate_to_language(english_response, original_lang)
        
        return jsonify({
            "response": final_response,
            "source": response_source,
            "confidence": float(confidence),
            "original_language": original_lang
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Chatbot API is running",
        "endpoints": {
            "/chat": "POST - submit {message}",
            "/health": "GET - health check"
        }
    })

if __name__ == '__main__':
    print("Starting Chatbot API Server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
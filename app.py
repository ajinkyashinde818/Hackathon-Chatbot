import gradio as gr
from chatbot_api import HybridChatbot

# Initialize the chatbot
chatbot = HybridChatbot()

def chat_api(message, language="en"):
    # Step 1: Translate to English
    english_question, original_lang = chatbot.translate_to_english(message)
    
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
    
    return final_response, response_source, confidence

# Define the Gradio interface
iface = gr.Interface(
    fn=chat_api,
    inputs=[gr.Textbox(label="Message"), gr.Textbox(label="Language", value="en")],
    outputs=[
        gr.Textbox(label="Response"),
        gr.Textbox(label="Source"),
        gr.Number(label="Confidence")
    ],
    title="Hybrid Chatbot",
    description="A hybrid chatbot using a local model and Gemini API for open-domain queries."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)    
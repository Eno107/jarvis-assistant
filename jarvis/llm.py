import os
import ollama
from dotenv import load_dotenv

load_dotenv()
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")

def process_command(text: str) -> str:
    """
    Process a text command using the local Ollama model.
    
    Args:
        text: The transcribed text command to process
        
    Returns:
        The model's response as a string
    """
    
    try:
        # Send the command to Ollama
        response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=[
            {
                'role': 'system',
                'content': 'You are Jarvis, a helpful AI assistant. Respond concisely and directly.'
            },
            {
                'role': 'user',
                'content': text
            }
        ])
        
        return response['message']['content']
    except Exception as e:
        print(f"[Jarvis] Error processing command with Ollama: {e}")
        return "Sorry, I encountered an error processing your command." 
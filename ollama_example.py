# ollama_example.py
# Query local Ollama models using the Ollama REST API (via requests)
# Author: CampusPe Assignment - Generative AI
#
# Prerequisites:
#   1. Install Ollama: https://ollama.ai/
#   2. Pull a model: ollama pull llama3
#   3. Start the server: ollama serve  (runs on http://localhost:11434)

import requests
import json

# ─── API Configuration ───────────────────────────────────────────────────────
# Ollama runs locally — no API key required!
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "llama3"   # Change to any model you have pulled


# ─── Query Function ───────────────────────────────────────────────────────────
def query_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Send a prompt to the local Ollama server and return the response text.

    Args:
        prompt : The user's input prompt.
        model  : The Ollama model to use (must be already pulled).

    Returns:
        The AI-generated response as a string.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model" : model,
        "prompt": prompt,
        "stream": False,  # Get full response at once
        "options": {
            "temperature": 0.7,
            "num_predict": 500,
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "No response received.").strip()

    except requests.exceptions.ConnectionError:
        return (
            "Error: Cannot connect to Ollama. "
            "Make sure Ollama is running ('ollama serve') on localhost:11434."
        )
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model may be loading — please try again."
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("     Ollama (Local) - Query Interface")
    print("=" * 50)
    print(f"Using model : {DEFAULT_MODEL}")
    print(f"Server URL  : {OLLAMA_BASE_URL}")

    user_prompt = input("\nEnter your prompt: ").strip()

    if not user_prompt:
        print("No prompt entered. Exiting.")
    else:
        print("\nQuerying Ollama (local)...\n")
        result = query_ollama(user_prompt)
        print("─" * 50)
        print("Response:")
        print("─" * 50)
        print(result)
        print("─" * 50)

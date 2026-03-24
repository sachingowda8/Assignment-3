# huggingface_example.py
# Query Hugging Face Inference API using the requests library
# Author: CampusPe Assignment - Generative AI

import os
import requests

# ─── API Configuration ───────────────────────────────────────────────────────
# Load API key from environment variable (never hardcode!)
api_key = os.getenv("HUGGINGFACE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "HUGGINGFACE_API_KEY not found. "
        "Please set it: $env:HUGGINGFACE_API_KEY='your-key-here'"
    )

# Default model — feel free to change to any text-generation model on HF Hub
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL = f"https://api-inference.huggingface.co/models/{DEFAULT_MODEL}"

HEADERS = {"Authorization": f"Bearer {api_key}"}


# ─── Query Function ───────────────────────────────────────────────────────────
def query_huggingface(prompt: str, max_new_tokens: int = 500) -> str:
    """
    Send a prompt to the Hugging Face Inference API and return the response text.

    Args:
        prompt        : The user's input prompt.
        max_new_tokens: Maximum tokens to generate in the response.

    Returns:
        The AI-generated response as a string.
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "return_full_text": False,  # Only return the generated part
        }
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()

        # Handle different response formats from HF
        if isinstance(data, list) and len(data) > 0:
            generated = data[0].get("generated_text", "")
            return generated.strip() if generated else "No response received."
        elif isinstance(data, dict):
            # Some models return an error dict
            if "error" in data:
                return f"API Error: {data['error']}"
        return str(data)

    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {e.response.status_code} - {e.response.text}"
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model may still be loading on HF — try again in 30s."
    except Exception as e:
        return f"Error querying Hugging Face: {str(e)}"


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   Hugging Face Inference API - Query Interface")
    print("=" * 55)
    print(f"Model: {DEFAULT_MODEL}")

    user_prompt = input("\nEnter your prompt: ").strip()

    if not user_prompt:
        print("No prompt entered. Exiting.")
    else:
        print("\nQuerying Hugging Face API...\n")
        result = query_huggingface(user_prompt)
        print("─" * 55)
        print("Response:")
        print("─" * 55)
        print(result)
        print("─" * 55)

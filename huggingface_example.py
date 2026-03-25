import os
import time
import requests

# 1. API Configuration
# Get access token from environment
api_key = os.getenv("HUGGINGFACE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] HUGGINGFACE_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:HUGGINGFACE_API_KEY = 'your-token-here'\n"
        "  Linux / Mac        : export HUGGINGFACE_API_KEY='your-token-here'\n"
        "  Get your token at  : https://huggingface.co/settings/tokens"
    )

# Using Mistral 7B Instruct v0.3 via the new Inference Router
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
API_URL = f"https://router.huggingface.co/hf-inference/models/{DEFAULT_MODEL}/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# Model Settings
MAX_TOKENS = 500
TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 5

def query_huggingface(prompt, max_tokens=MAX_TOKENS):
    """
    Queries Hugging Face Inference API using Chat Comprehensions format.
    Includes retry logic specifically for 503 (model loading) errors.
    """
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False # Inference Router supports streaming but implementation varies
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            
            # Handle rate limits or 503 (model still loading)
            if response.status_code == 503 or response.status_code == 429:
                wait = RETRY_DELAY * attempt
                print(f"  [Attempt {attempt}] Provider busy (Status {response.status_code}). Retrying in {wait}s...")
                time.sleep(wait)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip()
            elif "error" in data:
                return f"API Error: {data['error']}"
            return str(data)

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            return f"HTTP Error {status}: {e.response.text[:200]}"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return f"Error after {MAX_RETRIES} attempts: {str(e)}"
                
    return "Failed to get a response after multiple attempts."

if __name__ == "__main__":
    print("=" * 58)
    print("    Hugging Face Inference API - Query Interface")
    print("=" * 58)
    print(f"  Model      : {DEFAULT_MODEL}")
    print(f"  Max Tokens : {MAX_TOKENS}")
    print(f"  Temperature: {TEMPERATURE}")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 58)

    while True:
        user_prompt = input("\nEnter your prompt: ").strip()

        if not user_prompt:
            continue

        if user_prompt.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        print(f"\nQuerying Hugging Face ({DEFAULT_MODEL})...\n")
        result = query_huggingface(user_prompt)
        
        print("─" * 58)
        print("Response:")
        print("─" * 58)
        print(result)
        print("─" * 58)

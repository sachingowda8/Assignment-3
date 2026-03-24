import os
import time
import requests

api_key = os.getenv("HUGGINGFACE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] HUGGINGFACE_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:HUGGINGFACE_API_KEY = 'your-token-here'\n"
        "  Linux / Mac        : export HUGGINGFACE_API_KEY='your-token-here'\n"
        "  Get your token at  : https://huggingface.co/settings/tokens"
    )

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL = f"https://api-inference.huggingface.co/models/{DEFAULT_MODEL}"
HEADERS = {"Authorization": f"Bearer {api_key}"}

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 5

def query_huggingface(prompt, max_new_tokens=MAX_NEW_TOKENS):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": TEMPERATURE,
            "return_full_text": False,
            "do_sample": True,
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "").strip()
            elif isinstance(data, dict):
                if "error" in data and "loading" in data["error"].lower():
                    wait = min(int(data.get("estimated_time", RETRY_DELAY * attempt)), 30)
                    time.sleep(wait)
                    continue
                return str(data)
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return f"Error after {MAX_RETRIES} attempts: {str(e)}"
    return "Failed to get a response."

if __name__ == "__main__":
    print("=" * 55)
    print("    Hugging Face Inference API - Query Interface")
    print("=" * 55)
    while True:
        user_prompt = input("\nEnter your prompt: ").strip()
        if not user_prompt: continue
        if user_prompt.lower() in ("quit", "exit"): break
        print(f"\nQuerying Hugging Face...\n")
        print(f"Response:\n{query_huggingface(user_prompt)}")

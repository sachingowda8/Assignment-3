# huggingface_example.py
# Query Hugging Face Inference API using the requests library
# Author: CampusPe Assignment - Generative AI
#
# Setup:
#   pip install requests
#   $env:HUGGINGFACE_API_KEY = "your-token-here"
#   Get your token at: https://huggingface.co/settings/tokens

import os
import time
import requests

# ─── API Configuration ────────────────────────────────────────────────────────
# Load API token from environment variable (NEVER hardcode!)
api_key = os.getenv("HUGGINGFACE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] HUGGINGFACE_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:HUGGINGFACE_API_KEY = 'your-token-here'\n"
        "  Linux / Mac        : export HUGGINGFACE_API_KEY='your-token-here'\n"
        "  Get your token at  : https://huggingface.co/settings/tokens"
    )

# ─── Model Configuration ──────────────────────────────────────────────────────
# You can change this to any text-generation model on Hugging Face Hub
# Popular options:
#   - "mistralai/Mistral-7B-Instruct-v0.2"     (general chat)
#   - "google/flan-t5-large"                    (instruction following)
#   - "facebook/blenderbot-400M-distill"        (conversational)
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL       = f"https://api-inference.huggingface.co/models/{DEFAULT_MODEL}"
HEADERS       = {"Authorization": f"Bearer {api_key}"}

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 500    # Maximum tokens to generate
TEMPERATURE    = 0.7    # Creativity level
MAX_RETRIES    = 3      # Retry attempts on error (model loading, rate limit)
RETRY_DELAY    = 5      # Base wait in seconds — HF models can be slow to load


# ─── Query Function with Retry Logic ─────────────────────────────────────────
def query_huggingface(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Send a prompt to the Hugging Face Inference API and return the response text.
    Includes retry logic to handle model loading delays (cold starts).

    Args:
        prompt        : The user's input prompt.
        max_new_tokens: Maximum new tokens to generate.

    Returns:
        The AI-generated response as a string.
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens"  : max_new_tokens,
            "temperature"     : TEMPERATURE,
            "return_full_text": False,    # Only return the generated part, not the input
            "do_sample"       : True,     # Enable sampling for temperature to have effect
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            response.raise_for_status()

            data = response.json()

            # Handle different response formats from Hugging Face
            if isinstance(data, list) and len(data) > 0:
                generated = data[0].get("generated_text", "")
                if generated:
                    return generated.strip()
                return "No response text received from the model."

            elif isinstance(data, dict):
                # Model is still loading — wait and retry
                if "error" in data and "loading" in data["error"].lower():
                    estimated = data.get("estimated_time", RETRY_DELAY * attempt)
                    wait      = min(int(estimated), 30)   # Cap wait at 30 seconds
                    print(f"  [Model Loading] Waiting {wait}s for model to load (attempt {attempt}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue
                elif "error" in data:
                    return f"API Error: {data['error']}"

            return str(data)

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status == 503 and attempt < MAX_RETRIES:
                # 503 often means the model is loading
                wait = RETRY_DELAY * attempt
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Model unavailable (503). Retrying in {wait}s...")
                time.sleep(wait)
            elif status == 429 and attempt < MAX_RETRIES:
                # 429 means rate limit hit
                wait = RETRY_DELAY * attempt * 2
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Rate limit hit (429). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return f"HTTP Error {status}: {e.response.text}"

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Timed out. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return "Error: Request timed out. The model may be too large — try a smaller one."

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Error: {error_msg}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return f"Error querying Hugging Face after {MAX_RETRIES} attempts: {error_msg}"

    return "Failed to get a response after all retries."


# ─── Model Switcher Helper ────────────────────────────────────────────────────
def query_huggingface_model(prompt: str, model_name: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Query a specific Hugging Face model by name (instead of the default).

    Args:
        prompt        : The user's input prompt.
        model_name    : Full HF model name (e.g., 'google/flan-t5-large').
        max_new_tokens: Maximum new tokens to generate.

    Returns:
        The AI-generated response as a string.
    """
    url     = f"https://api-inference.huggingface.co/models/{model_name}"
    payload = {
        "inputs"    : prompt,
        "parameters": {
            "max_new_tokens"  : max_new_tokens,
            "temperature"     : TEMPERATURE,
            "return_full_text": False,
        }
    }
    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "No response.").strip()
        elif isinstance(data, dict) and "error" in data:
            return f"API Error: {data['error']}"
        return str(data)
    except Exception as e:
        return f"Error: {str(e)}"


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 58)
    print("    Hugging Face Inference API - Query Interface")
    print("=" * 58)
    print(f"  Model      : {DEFAULT_MODEL}")
    print(f"  Max Tokens : {MAX_NEW_TOKENS}")
    print(f"  Temperature: {TEMPERATURE}")
    print("  Type 'quit' or 'exit' to end the session.")
    print("  Type 'model <name>' to switch to a different HF model.")
    print("=" * 58)

    active_model = DEFAULT_MODEL

    while True:
        user_prompt = input("\nEnter your prompt: ").strip()

        if not user_prompt:
            print("  [!] Empty prompt — please enter something.")
            continue

        if user_prompt.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        # Switch model on the fly
        if user_prompt.lower().startswith("model "):
            new_model  = user_prompt[6:].strip()
            active_model = new_model
            global API_URL
            API_URL    = f"https://api-inference.huggingface.co/models/{new_model}"
            print(f"  [✓] Switched to model: {active_model}")
            continue

        print(f"\nQuerying Hugging Face ({active_model})...\n")
        result = (
            query_huggingface(user_prompt)
            if active_model == DEFAULT_MODEL
            else query_huggingface_model(user_prompt, active_model)
        )
        print("─" * 58)
        print("Response:")
        print("─" * 58)
        print(result)
        print("─" * 58)

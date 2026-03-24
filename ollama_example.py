# ollama_example.py
# Query local Ollama models using the Ollama REST API
# Author: CampusPe Assignment - Generative AI
#
# Prerequisites:
#   1. Install Ollama : https://ollama.ai/
#   2. Pull a model  : ollama pull llama3
#   3. Start the server: ollama serve   (runs on http://localhost:11434)
#
# No API key required — Ollama runs entirely on your local machine!

import requests
import json
import time

# ─── API Configuration ────────────────────────────────────────────────────────
# Ollama listens on localhost by default — no API key needed
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "llama3"   # Change to any model you have pulled (e.g., mistral, phi3)

# ─── Constants ────────────────────────────────────────────────────────────────
TEMPERATURE = 0.7    # Creativity level
NUM_PREDICT = 500    # Max tokens to generate
MAX_RETRIES = 3      # Retry attempts on failure
RETRY_DELAY = 3      # Base wait time in seconds (doubles each retry)

# ─── Conversation History ─────────────────────────────────────────────────────
# Build a running conversation string so Ollama remembers context
conversation_history = []


def _build_prompt_with_history(user_prompt: str) -> str:
    """Build a full prompt string including conversation history."""
    full_prompt = ""
    for entry in conversation_history:
        role  = entry["role"].capitalize()
        text  = entry["content"]
        full_prompt += f"{role}: {text}\n"
    full_prompt += f"User: {user_prompt}\nAssistant:"
    return full_prompt


# ─── Query Function with Retry Logic ─────────────────────────────────────────
def query_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Send a prompt to the local Ollama server and return the response text.
    Includes conversation history context and exponential retry logic.

    Args:
        prompt : The user's input prompt.
        model  : The Ollama model to use (must be already pulled).

    Returns:
        The AI-generated response as a string.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"

    # Include conversation history for context
    full_prompt = _build_prompt_with_history(prompt)

    payload = {
        "model"  : model,
        "prompt" : full_prompt,
        "stream" : False,     # Get the complete response at once
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            data  = response.json()
            reply = data.get("response", "No response received.").strip()

            # Save this exchange to conversation history
            conversation_history.append({"role": "user",      "content": prompt})
            conversation_history.append({"role": "assistant",  "content": reply})

            return reply

        except requests.exceptions.ConnectionError:
            return (
                "[ERROR] Cannot connect to Ollama.\n"
                "  Make sure Ollama is installed and running:\n"
                "    → Run: ollama serve\n"
                "    → Or start the Ollama desktop app"
            )
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Timed out. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return "Error: Request timed out. The model may be too large or still loading."
        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Error: {error_msg}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return f"Error querying Ollama after {MAX_RETRIES} attempts: {error_msg}"


# ─── Streaming Query Function (Bonus) ────────────────────────────────────────
def query_ollama_streaming(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Stream the Ollama response chunk-by-chunk for a typing effect.

    Args:
        prompt: The user's input prompt.
        model : Ollama model to use.

    Returns:
        The full response text after streaming completes.
    """
    url         = f"{OLLAMA_BASE_URL}/api/generate"
    full_prompt = _build_prompt_with_history(prompt)
    full_response = ""

    payload = {
        "model"  : model,
        "prompt" : full_prompt,
        "stream" : True,    # Enable streaming
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT}
    }

    try:
        print("Response (streaming):")
        print("─" * 50)
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    print(token, end="", flush=True)
                    full_response += token
                    if chunk.get("done"):
                        break
        print()
        print("─" * 50)

        conversation_history.append({"role": "user",     "content": prompt})
        conversation_history.append({"role": "assistant", "content": full_response})
        return full_response

    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot connect to Ollama. Is 'ollama serve' running?"
    except Exception as e:
        return f"Error during streaming: {str(e)}"


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("      Ollama (Local) - Query Interface")
    print("=" * 55)
    print(f"  Model      : {DEFAULT_MODEL}")
    print(f"  Server URL : {OLLAMA_BASE_URL}")
    print(f"  Temperature: {TEMPERATURE}")
    print("  Type 'quit' or 'exit' to end the session.")
    print("  Type 'stream' before your prompt to use streaming.")
    print("  Type 'history' to view conversation history.")
    print("=" * 55)

    while True:
        user_prompt = input("\nEnter your prompt: ").strip()

        if not user_prompt:
            print("  [!] Empty prompt — please enter something.")
            continue

        if user_prompt.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        # Show conversation history
        if user_prompt.lower() == "history":
            if not conversation_history:
                print("  No conversation history yet.")
            else:
                print("\n─── Conversation History ───")
                for i, msg in enumerate(conversation_history, 1):
                    print(f"  [{i}] {msg['role'].upper()}: {msg['content'][:80]}...")
            continue

        # --- Streaming mode ---
        if user_prompt.lower().startswith("stream "):
            actual_prompt = user_prompt[7:].strip()
            print(f"\nQuerying Ollama (streaming)...\n")
            query_ollama_streaming(actual_prompt)

        # --- Normal mode ---
        else:
            print("\nQuerying Ollama (local)...\n")
            result = query_ollama(user_prompt)
            print("─" * 55)
            print("Response:")
            print("─" * 55)
            print(result)
            print("─" * 55)

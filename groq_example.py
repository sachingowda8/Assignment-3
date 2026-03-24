# groq_example.py
# Query Groq Llama models using the Groq Python library
# Author: CampusPe Assignment - Generative AI
#
# Setup:
#   pip install groq
#   $env:GROQ_API_KEY = "your-key-here"
#   Get your key at: https://console.groq.com/

import os
import time
from groq import Groq

# ─── API Configuration ────────────────────────────────────────────────────────
# Load API key from environment variable (NEVER hardcode your key!)
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] GROQ_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:GROQ_API_KEY = 'your-key-here'\n"
        "  Linux / Mac        : export GROQ_API_KEY='your-key-here'\n"
        "  Get your key at    : https://console.groq.com/"
    )

# Initialize the Groq client
client = Groq(api_key=api_key)

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "llama3-8b-8192"   # Fast open-source Llama 3 via Groq
MAX_TOKENS    = 500                # Maximum response tokens
TEMPERATURE   = 0.7                # Creativity: 0 = focused, 1 = creative
MAX_RETRIES   = 3                  # Retry attempts for transient errors
RETRY_DELAY   = 2                  # Base wait time in seconds (doubles each retry)

# ─── Conversation History ─────────────────────────────────────────────────────
# Maintains multi-turn chat context so the model remembers previous messages
conversation_history = [
    {"role": "system", "content": "You are a helpful, knowledgeable, and friendly AI assistant."}
]


# ─── Query Function with Retry Logic ─────────────────────────────────────────
def query_groq(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = MAX_TOKENS) -> str:
    """
    Send a prompt to Groq Llama and return the response text.
    Includes conversation history (multi-turn) and exponential retry logic.

    Args:
        prompt    : The user's input prompt.
        model     : Groq model to use (default: llama3-8b-8192).
        max_tokens: Maximum tokens in the response.

    Returns:
        The AI-generated response as a string.
    """
    # Add user message to conversation history for multi-turn context
    conversation_history.append({"role": "user", "content": prompt})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=conversation_history,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
            )

            # Extract the text response
            reply = response.choices[0].message.content.strip()

            # Store assistant's reply in history
            conversation_history.append({"role": "assistant", "content": reply})

            return reply

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt   # Exponential back-off
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Error: {error_msg}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                conversation_history.pop()   # Remove unanswered message from history
                return f"Error querying Groq after {MAX_RETRIES} attempts: {error_msg}"


# ─── Streaming Query Function (Bonus) ────────────────────────────────────────
def query_groq_streaming(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Stream the Groq response token-by-token for a real-time typing effect.

    Args:
        prompt: The user's input prompt.
        model : Groq model to use.

    Returns:
        The full response text after streaming completes.
    """
    conversation_history.append({"role": "user", "content": prompt})
    full_response = ""

    try:
        # stream=True sends tokens as they are generated
        stream = client.chat.completions.create(
            model=model,
            messages=conversation_history,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )

        print("Response (streaming):")
        print("─" * 50)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                full_response += delta

        print()   # New line after streaming
        print("─" * 50)

        conversation_history.append({"role": "assistant", "content": full_response})
        return full_response

    except Exception as e:
        conversation_history.pop()
        return f"Error during streaming: {str(e)}"


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("        Groq Llama - Query Interface")
    print("=" * 55)
    print(f"  Model      : {DEFAULT_MODEL}")
    print(f"  Max Tokens : {MAX_TOKENS}")
    print(f"  Temperature: {TEMPERATURE}")
    print("  Type 'quit' or 'exit' to end the session.")
    print("  Type 'stream' before your prompt to use streaming.")
    print("=" * 55)

    while True:
        user_prompt = input("\nEnter your prompt: ").strip()

        if not user_prompt:
            print("  [!] Empty prompt — please enter something.")
            continue

        if user_prompt.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        # --- Streaming mode ---
        if user_prompt.lower().startswith("stream "):
            actual_prompt = user_prompt[7:].strip()
            print(f"\nQuerying Groq Llama (streaming)...\n")
            query_groq_streaming(actual_prompt)

        # --- Normal mode ---
        else:
            print("\nQuerying Groq Llama...\n")
            result = query_groq(user_prompt)
            print("─" * 55)
            print("Response:")
            print("─" * 55)
            print(result)
            print("─" * 55)

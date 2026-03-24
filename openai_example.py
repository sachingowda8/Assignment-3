# openai_example.py
# Query OpenAI GPT models using the OpenAI Python library
# Author: CampusPe Assignment - Generative AI
#
# Setup:
#   pip install openai
#   $env:OPENAI_API_KEY = "your-key-here"
#   Get your key at: https://platform.openai.com/api-keys

import os
import time
from openai import OpenAI

# ─── API Configuration ────────────────────────────────────────────────────────
# Load API key from environment variable (NEVER hardcode your key!)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] OPENAI_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:OPENAI_API_KEY = 'your-key-here'\n"
        "  Linux / Mac        : export OPENAI_API_KEY='your-key-here'\n"
        "  Get your key at    : https://platform.openai.com/api-keys"
    )

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=api_key)

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "gpt-3.5-turbo"   # Change to "gpt-4o" if your account supports it
MAX_TOKENS     = 500                # Maximum tokens in the response
TEMPERATURE    = 0.7                # Creativity level: 0 = deterministic, 1 = creative
MAX_RETRIES    = 3                  # Number of retry attempts on transient errors
RETRY_DELAY    = 2                  # Seconds to wait between retries (doubles each time)

# ─── Conversation History ─────────────────────────────────────────────────────
# Keeps track of the full conversation so GPT has context for follow-up questions
conversation_history = [
    {"role": "system", "content": "You are a helpful, knowledgeable, and friendly AI assistant."}
]


# ─── Query Function with Retry Logic ─────────────────────────────────────────
def query_openai(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = MAX_TOKENS) -> str:
    """
    Send a prompt to OpenAI GPT and return the response text.
    Includes conversation history (multi-turn chat) and retry logic.

    Args:
        prompt    : The user's input prompt.
        model     : GPT model to use (default: gpt-3.5-turbo).
        max_tokens: Maximum number of tokens in the response.

    Returns:
        The AI-generated response as a string.
    """
    # Add the new user message to the conversation history
    conversation_history.append({"role": "user", "content": prompt})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Send the full conversation history to maintain context
            response = client.chat.completions.create(
                model=model,
                messages=conversation_history,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
            )

            # Extract the assistant's reply
            reply = response.choices[0].message.content.strip()

            # Save assistant reply to history for next turn
            conversation_history.append({"role": "assistant", "content": reply})

            return reply

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt   # Exponential back-off
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Error: {error_msg}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                # Remove the unanswered user message from history
                conversation_history.pop()
                return f"Error querying OpenAI after {MAX_RETRIES} attempts: {error_msg}"


# ─── Streaming Query Function (Bonus) ────────────────────────────────────────
def query_openai_streaming(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Stream the OpenAI response token-by-token for a real-time typing effect.

    Args:
        prompt: The user's input prompt.
        model : GPT model to use.

    Returns:
        The full response text after streaming completes.
    """
    conversation_history.append({"role": "user", "content": prompt})
    full_response = ""

    try:
        # stream=True enables chunk-by-chunk delivery
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

        print()  # New line after streaming ends
        print("─" * 50)

        # Save to history
        conversation_history.append({"role": "assistant", "content": full_response})
        return full_response

    except Exception as e:
        conversation_history.pop()
        return f"Error during streaming: {str(e)}"


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("         OpenAI GPT - Query Interface")
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
            print(f"\nQuerying OpenAI GPT (streaming)...\n")
            query_openai_streaming(actual_prompt)

        # --- Normal mode ---
        else:
            print("\nQuerying OpenAI GPT...\n")
            result = query_openai(user_prompt)
            print("─" * 55)
            print("Response:")
            print("─" * 55)
            print(result)
            print("─" * 55)

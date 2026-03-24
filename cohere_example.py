# cohere_example.py
# Query Cohere models using the Cohere Python library
# Author: CampusPe Assignment - Generative AI
#
# Setup:
#   pip install cohere
#   $env:COHERE_API_KEY = "your-key-here"
#   Get your key at: https://dashboard.cohere.com/

import os
import time
import cohere

# ─── API Configuration ────────────────────────────────────────────────────────
# Load API key from environment variable (NEVER hardcode!)
api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] COHERE_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:COHERE_API_KEY = 'your-key-here'\n"
        "  Linux / Mac        : export COHERE_API_KEY='your-key-here'\n"
        "  Get your key at    : https://dashboard.cohere.com/"
    )

# Initialize the Cohere client (v2 API)
co = cohere.ClientV2(api_key=api_key)

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "command-r-plus"   # Best for chat; alternatives: "command-r", "command"
MAX_TOKENS    = 500                # Maximum response tokens
TEMPERATURE   = 0.7                # Creativity level
MAX_RETRIES   = 3                  # Retry attempts on transient errors
RETRY_DELAY   = 2                  # Base wait in seconds (doubles each retry)

# ─── Conversation History ─────────────────────────────────────────────────────
# Maintains multi-turn chat context for the Cohere model
conversation_history = []   # List of {"role": "user"/"assistant", "content": "..."}


# ─── Query Function with Retry Logic ─────────────────────────────────────────
def query_cohere(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = MAX_TOKENS) -> str:
    """
    Send a prompt to Cohere and return the response text.
    Includes conversation history (multi-turn) and exponential retry logic.

    Args:
        prompt    : The user's input prompt.
        model     : Cohere model to use (default: command-r-plus).
        max_tokens: Maximum tokens to generate in the response.

    Returns:
        The AI-generated response as a string.
    """
    # Add user message to conversation history for multi-turn context
    conversation_history.append({"role": "user", "content": prompt})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Send full conversation history for contextual responses
            response = co.chat(
                model=model,
                messages=conversation_history,
                max_tokens=max_tokens,
            )

            # Extract the text response from Cohere's response object
            reply = response.message.content[0].text.strip()

            # Save assistant reply to history
            conversation_history.append({"role": "assistant", "content": reply})

            return reply

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt   # Exponential back-off
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Error: {error_msg}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                conversation_history.pop()   # Remove unanswered message
                return f"Error querying Cohere after {MAX_RETRIES} attempts: {error_msg}"


# ─── Streaming Query Function (Bonus) ────────────────────────────────────────
def query_cohere_streaming(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Stream the Cohere response token-by-token for a real-time typing effect.

    Args:
        prompt: The user's input prompt.
        model : Cohere model to use.

    Returns:
        The full response text after streaming completes.
    """
    conversation_history.append({"role": "user", "content": prompt})
    full_response = ""

    try:
        print("Response (streaming):")
        print("─" * 50)

        # stream=True sends tokens as they are generated
        with co.chat_stream(
            model=model,
            messages=conversation_history,
            max_tokens=MAX_TOKENS,
        ) as stream:
            for event in stream:
                # Text delta events contain individual tokens
                if event.type == "content-delta":
                    token = event.delta.message.content.text
                    print(token, end="", flush=True)
                    full_response += token

        print()
        print("─" * 50)

        conversation_history.append({"role": "assistant", "content": full_response})
        return full_response

    except Exception as e:
        conversation_history.pop()
        return f"Error during streaming: {str(e)}"


# ─── Show History ─────────────────────────────────────────────────────────────
def show_history():
    """Display the current conversation history."""
    if not conversation_history:
        print("  No conversation history yet.")
        return
    print("\n─── Conversation History ───")
    for i, msg in enumerate(conversation_history, 1):
        preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        print(f"  [{i}] {msg['role'].upper()}: {preview}")


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("         Cohere - Query Interface")
    print("=" * 55)
    print(f"  Model      : {DEFAULT_MODEL}")
    print(f"  Max Tokens : {MAX_TOKENS}")
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
            show_history()
            continue

        # --- Streaming mode ---
        if user_prompt.lower().startswith("stream "):
            actual_prompt = user_prompt[7:].strip()
            print(f"\nQuerying Cohere (streaming)...\n")
            query_cohere_streaming(actual_prompt)

        # --- Normal mode ---
        else:
            print("\nQuerying Cohere...\n")
            result = query_cohere(user_prompt)
            print("─" * 55)
            print("Response:")
            print("─" * 55)
            print(result)
            print("─" * 55)

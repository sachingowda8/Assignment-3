import os
import time
from groq import Groq

# 1. API Configuration
# Load Groq API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] GROQ_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:GROQ_API_KEY = 'your-key-here'\n"
        "  Linux / Mac        : export GROQ_API_KEY='your-key-here'\n"
        "  Get your key at    : https://console.groq.com/"
    )

# Initialize Groq client
client = Groq(api_key=api_key)

# Default Model Parameters
# Note: Using Llama 3.3 70B for high-quality responses
DEFAULT_MODEL = "llama-3.3-70b-versatile" 
MAX_TOKENS = 500
TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 2

# Conversation history to maintain context during chat
conversation_history = [
    {"role": "system", "content": "You are a helpful, knowledgeable, and friendly AI assistant."}
]

def query_groq(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS):
    """
    Queries the Groq API with persistent history and basic retry logic.
    """
    conversation_history.append({"role": "user", "content": prompt})

    # Retry logic handles transient network or rate limit errors
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=conversation_history,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
            )

            reply = response.choices[0].message.content.strip()
            conversation_history.append({"role": "assistant", "content": reply})
            return reply

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Error: {error_msg}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                conversation_history.pop() # Clean history on failure
                return f"Error querying Groq after {MAX_RETRIES} attempts: {error_msg}"

def query_groq_streaming(prompt, model=DEFAULT_MODEL):
    """
    Streams response from Groq token-by-token for better UX.
    """
    conversation_history.append({"role": "user", "content": prompt})
    full_response = ""

    try:
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

        print()
        print("─" * 50)

        conversation_history.append({"role": "assistant", "content": full_response})
        return full_response

    except Exception as e:
        conversation_history.pop()
        return f"Error during streaming: {str(e)}"

if __name__ == "__main__":
    # CLI Interface
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
            continue

        if user_prompt.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        if user_prompt.lower().startswith("stream "):
            actual_prompt = user_prompt[7:].strip()
            print(f"\nQuerying Groq Llama (streaming)...\n")
            query_groq_streaming(actual_prompt)
        else:
            print("\nQuerying Groq Llama...\n")
            result = query_groq(user_prompt)
            print("─" * 55)
            print("Response:")
            print("─" * 55)
            print(result)
            print("─" * 55)

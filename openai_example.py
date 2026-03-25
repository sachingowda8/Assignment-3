import os
import time
from openai import OpenAI

# 1. API Configuration
# Load API key from environment variables for security (never hardcode keys)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] OPENAI_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:OPENAI_API_KEY = 'your-key-here'\n"
        "  Linux / Mac        : export OPENAI_API_KEY='your-key-here'\n"
        "  Get your key at    : https://platform.openai.com/api-keys"
    )

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Default Model Parameters
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 500
TEMPERATURE = 0.7
MAX_RETRIES = 3      # Number of retry attempts for API errors
RETRY_DELAY = 2      # Base delay in seconds for exponential backoff

# Initialize conversation history to maintain context
conversation_history = [
    {"role": "system", "content": "You are a helpful, knowledgeable, and friendly AI assistant."}
]

def query_openai(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS):
    """
    Sends a prompt to OpenAI API and returns the generated response.
    Includes persistent conversation history and retry logic.
    """
    # Append the new user prompt to history
    conversation_history.append({"role": "user", "content": prompt})

    # Implement Retry logic with exponential backoff
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=conversation_history,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
            )

            # Extract the text reply
            reply = response.choices[0].message.content.strip()
            
            # Store the assistant's response in history for future context
            conversation_history.append({"role": "assistant", "content": reply})
            
            return reply

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Error: {error_msg}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                # Remove the failed prompt from history to keep it clean
                conversation_history.pop()
                return f"Error querying OpenAI after {MAX_RETRIES} attempts: {error_msg}"

def query_openai_streaming(prompt, model=DEFAULT_MODEL):
    """
    Version of query_openai that streams tokens to the console in real-time.
    """
    conversation_history.append({"role": "user", "content": prompt})
    full_response = ""

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=conversation_history,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True, # Enable streaming mode
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

        # Log assistant response to history
        conversation_history.append({"role": "assistant", "content": full_response})
        return full_response

    except Exception as e:
        conversation_history.pop()
        return f"Error during streaming: {str(e)}"

if __name__ == "__main__":
    # Main User Interface Loop
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

        # Check for streaming mode
        if user_prompt.lower().startswith("stream "):
            actual_prompt = user_prompt[7:].strip()
            print(f"\nQuerying OpenAI GPT (streaming)...\n")
            query_openai_streaming(actual_prompt)
        else:
            print("\nQuerying OpenAI GPT...\n")
            result = query_openai(user_prompt)
            print("─" * 55)
            print("Response:")
            print("─" * 55)
            print(result)
            print("─" * 55)

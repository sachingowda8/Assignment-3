import os
import time
import cohere

# 1. API Configuration
# Initialize Cohere Client from environment variable
api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] COHERE_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:COHERE_API_KEY = 'your-key-here'\n"
        "  Linux / Mac        : export COHERE_API_KEY='your-key-here'\n"
        "  Get your key at    : https://dashboard.cohere.com/"
    )

# Using the newest Cohere V2 Client
co = cohere.ClientV2(api_key=api_key)

# Default Model Selection
# 'command-r7b-12-2024' is the successor to the removed command-r-plus
DEFAULT_MODEL = "command-r7b-12-2024" 
MAX_TOKENS = 500
TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 2

# History management list
conversation_history = []

def query_cohere(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS):
    """
    Sends a query to Cohere using the V2 chat interface.
    """
    conversation_history.append({"role": "user", "content": prompt})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = co.chat(
                model=model,
                messages=conversation_history,
                max_tokens=max_tokens,
            )
            
            # v2 response structure uses .message.content[0].text
            reply = response.message.content[0].text.strip()
            
            conversation_history.append({"role": "assistant", "content": reply})
            return reply

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                conversation_history.pop() # Clean history on failure
                return f"Error after {MAX_RETRIES} attempts: {str(e)}"

def query_cohere_streaming(prompt, model=DEFAULT_MODEL):
    """
    Uses the Cohere V2 stream_chat context manager for real-time tokens.
    """
    conversation_history.append({"role": "user", "content": prompt})
    full_response = ""

    try:
        print("Response (streaming):")
        print("─" * 50)
        # Using the v2 chat_stream generator
        with co.chat_stream(model=model, messages=conversation_history, max_tokens=MAX_TOKENS) as stream:
            for event in stream:
                # Check for content tokens in the event stream
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
        return f"Streaming error: {str(e)}"

if __name__ == "__main__":
    print("=" * 55)
    print("         Cohere - Query Interface")
    print("=" * 55)
    print(f"  Model      : {DEFAULT_MODEL}")
    print("  Commands   : 'quit', 'exit', 'history', 'stream <prompt>'")
    print("=" * 55)

    while True:
        user_prompt = input("\nEnter your prompt: ").strip()

        if not user_prompt:
            continue

        if user_prompt.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        if user_prompt.lower() == "history":
            print("\nConversation History:")
            for msg in conversation_history:
                print(f"  {msg['role'].upper()}: {msg['content'][:80]}...")
            continue

        if user_prompt.lower().startswith("stream "):
            query_cohere_streaming(user_prompt[7:].strip())
        else:
            print(f"\nQuerying Cohere...\n")
            result = query_cohere(user_prompt)
            print("─" * 55)
            print("Response:")
            print("─" * 55)
            print(result)
            print("─" * 55)

import os
import time
from google import genai
from google.genai import types

# 1. API Configuration
# Get key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] GOOGLE_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:GOOGLE_API_KEY = 'your-key-here'\n"
        "  Linux / Mac        : export GOOGLE_API_KEY='your-key-here'\n"
        "  Get your key at    : https://aistudio.google.com/app/apikey"
    )

# Initialize the new Google GenAI Client
client = genai.Client(api_key=api_key)

# Default Model Parameters
DEFAULT_MODEL = "gemini-2.0-flash" # High-speed, high-quality multimodal model
MAX_TOKENS = 500
TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 2

# Chat history to maintain state
conversation_history = []

def query_gemini(prompt):
    """
    Queries Google Gemini using the contents-based stateful API.
    """
    # Append the new message to history in the required structure
    conversation_history.append({"role": "user", "parts": [{"text": prompt}]})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Generate content based on the full conversation history
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=conversation_history,
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
            )
            
            reply = response.text.strip()
            
            # Log assistant's part back into history
            conversation_history.append({"role": "model", "parts": [{"text": reply}]})
            return reply

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                conversation_history.pop() # Clean last failed prompt
                return f"Error after {MAX_RETRIES} attempts: {str(e)}"

def query_gemini_streaming(prompt):
    """
    Streams content from Gemini using the stream-enabled client method.
    """
    full_response = ""
    try:
        print("Response (streaming):")
        print("─" * 50)
        # Note: Streaming in 2-way chat requires special history handling, 
        # so we do a one-off completion here for demo purposes.
        for chunk in client.models.generate_content_stream(
            model=DEFAULT_MODEL,
            contents=prompt
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        print()
        print("─" * 50)
        return full_response
    except Exception as e:
        return f"Streaming error: {str(e)}"

if __name__ == "__main__":
    print("=" * 55)
    print("       Google Gemini - Query Interface")
    print("=" * 55)
    print(f"  Model      : {DEFAULT_MODEL}")
    print(f"  Max Tokens : {MAX_TOKENS}")
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
                # msg contains 'role' and 'parts' [ { 'text': '...' } ]
                print(f"  {msg['role'].upper()}: {msg['parts'][0]['text'][:80]}...")
            continue

        if user_prompt.lower().startswith("stream "):
            query_gemini_streaming(user_prompt[7:].strip())
        else:
            print(f"\nQuerying Google Gemini...\n")
            result = query_gemini(user_prompt)
            print("─" * 55)
            print("Response:")
            print("─" * 55)
            print(result)
            print("─" * 55)

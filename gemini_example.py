import os
import time
from google import genai
from google.genai import types

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] GOOGLE_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:GOOGLE_API_KEY = 'your-key-here'\n"
        "  Linux / Mac        : export GOOGLE_API_KEY='your-key-here'\n"
        "  Get your key at    : https://aistudio.google.com/app/apikey"
    )

client = genai.Client(api_key=api_key)

DEFAULT_MODEL = "gemini-2.0-flash"
MAX_TOKENS = 500
TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 2

conversation_history = []

def query_gemini(prompt):
    conversation_history.append({"role": "user", "parts": [{"text": prompt}]})
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=conversation_history,
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
            )
            reply = response.text.strip()
            conversation_history.append({"role": "model", "parts": [{"text": reply}]})
            return reply
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                conversation_history.pop()
                return f"Error after {MAX_RETRIES} attempts: {str(e)}"

def query_gemini_streaming(prompt):
    full_response = ""
    try:
        print("Response (streaming):")
        for chunk in client.models.generate_content_stream(
            model=DEFAULT_MODEL,
            contents=prompt
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        print()
        return full_response
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    print("=" * 55)
    print("       Google Gemini - Query Interface")
    print("=" * 55)
    print(f"  Model      : {DEFAULT_MODEL}")
    print(f"  Max Tokens : {MAX_TOKENS}")
    print(f"  Temperature: {TEMPERATURE}")
    print("  Type 'quit' or 'exit' to end the session.")
    print("  Type 'stream' before your prompt to use streaming.")
    print("=" * 55)

    while True:
        user_prompt = input("\nEnter your prompt: ").strip()
        if not user_prompt: continue
        if user_prompt.lower() in ("quit", "exit"): break
        if user_prompt.lower() == "history":
            for msg in conversation_history:
                print(f"{msg['role'].upper()}: {msg['parts'][0]['text'][:80]}...")
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

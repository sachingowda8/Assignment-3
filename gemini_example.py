import os
import time
import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] GOOGLE_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:GOOGLE_API_KEY = 'your-key-here'\n"
        "  Linux / Mac        : export GOOGLE_API_KEY='your-key-here'\n"
        "  Get your key at    : https://makersuite.google.com/app/apikey"
    )

genai.configure(api_key=api_key)

DEFAULT_MODEL = "gemini-1.5-flash"

GENERATION_CONFIG = genai.types.GenerationConfig(
    max_output_tokens=500,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
)

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

MAX_RETRIES = 3
RETRY_DELAY = 2

model = genai.GenerativeModel(
    DEFAULT_MODEL,
    generation_config=GENERATION_CONFIG,
    safety_settings=SAFETY_SETTINGS,
)
chat_session = model.start_chat(history=[])

def query_gemini(prompt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = chat_session.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return f"Error after {MAX_RETRIES} attempts: {str(e)}"

def query_gemini_streaming(prompt):
    full_response = ""
    try:
        stream = model.generate_content(prompt, stream=True)
        print("Response (streaming):")
        for chunk in stream:
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
    while True:
        user_prompt = input("\nEnter your prompt: ").strip()
        if not user_prompt: continue
        if user_prompt.lower() in ("quit", "exit"): break
        if user_prompt.lower() == "history":
            for msg in chat_session.history:
                print(f"{msg.role.upper()}: {msg.parts[0].text[:80]}...")
            continue
        if user_prompt.lower().startswith("stream "):
            query_gemini_streaming(user_prompt[7:].strip())
        else:
            print(f"Response:\n{query_gemini(user_prompt)}")

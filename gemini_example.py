# gemini_example.py
# Query Google Gemini using the google-generativeai library
# Author: CampusPe Assignment - Generative AI
#
# Setup:
#   pip install google-generativeai
#   $env:GOOGLE_API_KEY = "your-key-here"
#   Get your key at: https://makersuite.google.com/app/apikey

import os
import time
import google.generativeai as genai

# ─── API Configuration ────────────────────────────────────────────────────────
# Load API key from environment variable (NEVER hardcode!)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "\n[ERROR] GOOGLE_API_KEY environment variable not set.\n"
        "  Windows PowerShell : $env:GOOGLE_API_KEY = 'your-key-here'\n"
        "  Linux / Mac        : export GOOGLE_API_KEY='your-key-here'\n"
        "  Get your key at    : https://makersuite.google.com/app/apikey"
    )

# Configure the Gemini library with your API key
genai.configure(api_key=api_key)

# ─── Model Configuration ──────────────────────────────────────────────────────
# Available models: "gemini-1.5-flash" (fast, free tier)
#                   "gemini-1.5-pro"   (more capable, higher quota)
DEFAULT_MODEL = "gemini-1.5-flash"

GENERATION_CONFIG = genai.types.GenerationConfig(
    max_output_tokens=500,   # Maximum tokens to generate
    temperature=0.7,         # Creativity: 0 = precise, 1 = creative
    top_p=0.9,               # Nucleus sampling probability
    top_k=40,                # Top-K sampling
)

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",       "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",      "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT","threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_RETRIES = 3      # Retry attempts on transient errors
RETRY_DELAY = 2      # Base wait time in seconds (doubles each retry)

# ─── Initialize Model and Chat Session ───────────────────────────────────────
# Use a chat session to automatically maintain conversation history
model = genai.GenerativeModel(
    DEFAULT_MODEL,
    generation_config=GENERATION_CONFIG,
    safety_settings=SAFETY_SETTINGS,
)
chat_session = model.start_chat(history=[])


# ─── Query Function with Retry Logic ─────────────────────────────────────────
def query_gemini(prompt: str) -> str:
    """
    Send a prompt to Google Gemini and return the response text.
    Uses a persistent chat session to automatically maintain conversation history.
    Includes exponential retry logic for transient errors.

    Args:
        prompt: The user's input prompt.

    Returns:
        The AI-generated response as a string.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # send_message automatically adds to chat history
            response = chat_session.send_message(prompt)
            return response.text.strip()

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt   # Exponential back-off
                print(f"  [Retry {attempt}/{MAX_RETRIES}] Error: {error_msg}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return f"Error querying Gemini after {MAX_RETRIES} attempts: {error_msg}"


# ─── Streaming Query Function (Bonus) ────────────────────────────────────────
def query_gemini_streaming(prompt: str) -> str:
    """
    Stream the Gemini response chunk-by-chunk for a typing effect.

    Args:
        prompt: The user's input prompt.

    Returns:
        The full response text after streaming completes.
    """
    full_response = ""
    try:
        # stream=True sends chunks as they are generated
        stream = model.generate_content(prompt, stream=True)

        print("Response (streaming):")
        print("─" * 50)
        for chunk in stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        print()
        print("─" * 50)
        return full_response

    except Exception as e:
        return f"Error during streaming: {str(e)}"


# ─── Show Chat History ────────────────────────────────────────────────────────
def show_history():
    """Display the conversation history stored in the Gemini chat session."""
    history = chat_session.history
    if not history:
        print("  No conversation history yet.")
        return
    print("\n─── Conversation History ───")
    for i, message in enumerate(history, 1):
        role    = message.role.upper()
        content = message.parts[0].text[:80] + "..." if len(message.parts[0].text) > 80 else message.parts[0].text
        print(f"  [{i}] {role}: {content}")


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("       Google Gemini - Query Interface")
    print("=" * 55)
    print(f"  Model      : {DEFAULT_MODEL}")
    print(f"  Max Tokens : {GENERATION_CONFIG.max_output_tokens}")
    print(f"  Temperature: {GENERATION_CONFIG.temperature}")
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
            print(f"\nQuerying Google Gemini (streaming)...\n")
            query_gemini_streaming(actual_prompt)

        # --- Normal mode ---
        else:
            print("\nQuerying Google Gemini...\n")
            result = query_gemini(user_prompt)
            print("─" * 55)
            print("Response:")
            print("─" * 55)
            print(result)
            print("─" * 55)

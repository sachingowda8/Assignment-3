# gemini_example.py
# Query Google Gemini using the google-generativeai library
# Author: CampusPe Assignment - Generative AI

import os
import google.generativeai as genai

# ─── API Configuration ───────────────────────────────────────────────────────
# Load API key from environment variable (never hardcode!)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. "
        "Please set it: $env:GOOGLE_API_KEY='your-key-here'\n"
        "Get your key at: https://makersuite.google.com/app/apikey"
    )

# Configure the Gemini client
genai.configure(api_key=api_key)

# Initialize the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")


# ─── Query Function ───────────────────────────────────────────────────────────
def query_gemini(prompt: str) -> str:
    """
    Send a prompt to Google Gemini and return the response text.

    Args:
        prompt: The user's input prompt.

    Returns:
        The AI-generated response as a string.
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.7,
            )
        )
        return response.text.strip()

    except Exception as e:
        return f"Error querying Gemini: {str(e)}"


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   Google Gemini - Query Interface")
    print("=" * 50)

    user_prompt = input("\nEnter your prompt: ").strip()

    if not user_prompt:
        print("No prompt entered. Exiting.")
    else:
        print("\nQuerying Google Gemini...\n")
        result = query_gemini(user_prompt)
        print("─" * 50)
        print("Response:")
        print("─" * 50)
        print(result)
        print("─" * 50)

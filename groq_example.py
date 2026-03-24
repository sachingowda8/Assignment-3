# groq_example.py
# Query Groq Llama models using the Groq Python library
# Author: CampusPe Assignment - Generative AI

import os
from groq import Groq

# ─── API Configuration ───────────────────────────────────────────────────────
# Load API key from environment variable (never hardcode!)
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY not found. "
        "Please set it: $env:GROQ_API_KEY='your-key-here'"
    )

# Initialize the Groq client
client = Groq(api_key=api_key)


# ─── Query Function ───────────────────────────────────────────────────────────
def query_groq(prompt: str, model: str = "llama3-8b-8192", max_tokens: int = 500) -> str:
    """
    Send a prompt to Groq Llama and return the response text.

    Args:
        prompt    : The user's input prompt.
        model     : The Groq model to use (default: llama3-8b-8192).
        max_tokens: Maximum number of tokens in the response.

    Returns:
        The AI-generated response as a string.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error querying Groq: {str(e)}"


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("      Groq Llama - Query Interface")
    print("=" * 50)

    user_prompt = input("\nEnter your prompt: ").strip()

    if not user_prompt:
        print("No prompt entered. Exiting.")
    else:
        print("\nQuerying Groq Llama...\n")
        result = query_groq(user_prompt)
        print("─" * 50)
        print("Response:")
        print("─" * 50)
        print(result)
        print("─" * 50)

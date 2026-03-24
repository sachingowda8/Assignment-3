# openai_example.py
# Query OpenAI GPT models using the OpenAI Python library
# Author: CampusPe Assignment - Generative AI

import os
from openai import OpenAI

# ─── API Configuration ───────────────────────────────────────────────────────
# Load API key from environment variable (never hardcode!)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY not found. "
        "Please set it: $env:OPENAI_API_KEY='your-key-here'"
    )

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)


# ─── Query Function ───────────────────────────────────────────────────────────
def query_openai(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 500) -> str:
    """
    Send a prompt to OpenAI GPT and return the response text.

    Args:
        prompt    : The user's input prompt.
        model     : The GPT model to use (default: gpt-3.5-turbo).
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
        return f"Error querying OpenAI: {str(e)}"


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("        OpenAI GPT - Query Interface")
    print("=" * 50)

    user_prompt = input("\nEnter your prompt: ").strip()

    if not user_prompt:
        print("No prompt entered. Exiting.")
    else:
        print("\nQuerying OpenAI GPT...\n")
        result = query_openai(user_prompt)
        print("─" * 50)
        print("Response:")
        print("─" * 50)
        print(result)
        print("─" * 50)

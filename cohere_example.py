# cohere_example.py
# Query Cohere models using the Cohere Python library
# Author: CampusPe Assignment - Generative AI

import os
import cohere

# ─── API Configuration ───────────────────────────────────────────────────────
# Load API key from environment variable (never hardcode!)
api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise EnvironmentError(
        "COHERE_API_KEY not found. "
        "Please set it: $env:COHERE_API_KEY='your-key-here'\n"
        "Get your key at: https://dashboard.cohere.com/"
    )

# Initialize the Cohere client
co = cohere.ClientV2(api_key=api_key)


# ─── Query Function ───────────────────────────────────────────────────────────
def query_cohere(prompt: str, model: str = "command-r-plus", max_tokens: int = 500) -> str:
    """
    Send a prompt to Cohere and return the response text.

    Args:
        prompt    : The user's input prompt.
        model     : The Cohere model to use (default: command-r-plus).
        max_tokens: Maximum tokens to generate.

    Returns:
        The AI-generated response as a string.
    """
    try:
        response = co.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.message.content[0].text.strip()

    except Exception as e:
        return f"Error querying Cohere: {str(e)}"


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("      Cohere - Query Interface")
    print("=" * 50)

    user_prompt = input("\nEnter your prompt: ").strip()

    if not user_prompt:
        print("No prompt entered. Exiting.")
    else:
        print("\nQuerying Cohere...\n")
        result = query_cohere(user_prompt)
        print("─" * 50)
        print("Response:")
        print("─" * 50)
        print(result)
        print("─" * 50)

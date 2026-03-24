# multi_api_query.py
# Unified multi-provider AI query program (BONUS)
# Author: CampusPe Assignment - Generative AI
#
# Allows the user to select an AI provider, enter a prompt,
# and receive a response from the chosen provider.

import os
import sys

# ─── Provider imports (handled gracefully if not installed) ───────────────────

def import_providers():
    """Attempt to import each provider library; track availability."""
    available = {}

    try:
        from openai import OpenAI
        available["openai"] = OpenAI
    except ImportError:
        available["openai"] = None

    try:
        from groq import Groq
        available["groq"] = Groq
    except ImportError:
        available["groq"] = None

    try:
        import requests
        available["ollama"] = requests
    except ImportError:
        available["ollama"] = None

    try:
        import requests
        available["huggingface"] = requests
    except ImportError:
        available["huggingface"] = None

    try:
        import google.generativeai as genai
        available["gemini"] = genai
    except ImportError:
        available["gemini"] = None

    try:
        import cohere
        available["cohere"] = cohere
    except ImportError:
        available["cohere"] = None

    return available


# ─── Individual query functions ───────────────────────────────────────────────

def query_openai(prompt: str) -> str:
    """Query OpenAI GPT-3.5-turbo."""
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not set."
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI Error: {str(e)}"


def query_groq(prompt: str) -> str:
    """Query Groq Llama models."""
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environment variable not set."
    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq Error: {str(e)}"


def query_ollama(prompt: str) -> str:
    """Query local Ollama server."""
    import requests
    url = "http://localhost:11434/api/generate"
    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "No response.").strip()
    except requests.exceptions.ConnectionError:
        return "Error: Ollama not running. Start with: ollama serve"
    except Exception as e:
        return f"Ollama Error: {str(e)}"


def query_huggingface(prompt: str) -> str:
    """Query Hugging Face Inference API."""
    import requests
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        return "Error: HUGGINGFACE_API_KEY environment variable not set."
    model   = "mistralai/Mistral-7B-Instruct-v0.2"
    url     = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 500, "return_full_text": False}}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data[0].get("generated_text", "No response.").strip()
        return str(data)
    except Exception as e:
        return f"Hugging Face Error: {str(e)}"


def query_gemini(prompt: str) -> str:
    """Query Google Gemini."""
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY environment variable not set."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {str(e)}"


def query_cohere(prompt: str) -> str:
    """Query Cohere."""
    import cohere
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return "Error: COHERE_API_KEY environment variable not set."
    co = cohere.ClientV2(api_key=api_key)
    try:
        response = co.chat(
            model="command-r-plus",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response.message.content[0].text.strip()
    except Exception as e:
        return f"Cohere Error: {str(e)}"


# ─── Provider registry ────────────────────────────────────────────────────────

PROVIDERS = {
    "1": ("OpenAI GPT",           query_openai),
    "2": ("Groq Llama",           query_groq),
    "3": ("Ollama (Local)",       query_ollama),
    "4": ("Hugging Face",         query_huggingface),
    "5": ("Google Gemini",        query_gemini),
    "6": ("Cohere",               query_cohere),
}


# ─── Main Execution ───────────────────────────────────────────────────────────

def display_menu():
    """Display the provider selection menu."""
    print("\n" + "=" * 55)
    print("       Multi-API Query Tool - Select Provider")
    print("=" * 55)
    for key, (name, _) in PROVIDERS.items():
        print(f"  [{key}] {name}")
    print("  [q] Quit")
    print("=" * 55)


def main():
    print("\n🤖 Welcome to the Multi-AI API Query Tool!")
    print("   Query any AI provider from one program.\n")

    while True:
        display_menu()
        choice = input("\nSelect a provider (1-6 or q): ").strip().lower()

        if choice == "q":
            print("\nGoodbye! 👋")
            break

        if choice not in PROVIDERS:
            print("❌ Invalid choice. Please enter 1-6 or q.")
            continue

        provider_name, query_func = PROVIDERS[choice]
        print(f"\n✅ Selected: {provider_name}")

        user_prompt = input("Enter your prompt: ").strip()
        if not user_prompt:
            print("⚠️  No prompt entered. Returning to menu.")
            continue

        print(f"\n🔄 Querying {provider_name}...\n")
        result = query_func(user_prompt)

        print("─" * 55)
        print(f"📝 Response from {provider_name}:")
        print("─" * 55)
        print(result)
        print("─" * 55)

        again = input("\nQuery another provider? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye! 👋")
            break


if __name__ == "__main__":
    main()

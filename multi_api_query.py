# multi_api_query.py
# Multi-API Query Program — Query multiple AI providers from one interface
# Author: CampusPe Assignment - Generative AI
# Bonus Task: Unified interface with provider selection, streaming, comparison mode
#
# Setup:
#   pip install openai groq requests google-generativeai cohere
#   Set environment variables for the APIs you want to use.

import os
import sys
import time

# ─── Available Providers ──────────────────────────────────────────────────────
PROVIDERS = {
    "1": "OpenAI",
    "2": "Groq",
    "3": "Ollama",
    "4": "Hugging Face",
    "5": "Gemini",
    "6": "Cohere",
    "7": "Compare All",   # Bonus: query all providers at once
}


# ─── Lazy Client Initialization ───────────────────────────────────────────────
# Clients are initialized only when the user selects that provider,
# so missing API keys for other providers won't cause crashes.

def _init_openai():
    """Initialize and return the OpenAI client."""
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


def _init_groq():
    """Initialize and return the Groq client."""
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")
    return Groq(api_key=api_key)


def _init_gemini():
    """Initialize and return the Gemini model."""
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def _init_cohere():
    """Initialize and return the Cohere client."""
    import cohere
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError("COHERE_API_KEY not set.")
    return cohere.ClientV2(api_key=api_key)


# ─── Individual Query Functions ───────────────────────────────────────────────

def query_openai(prompt: str, history: list) -> str:
    """Query OpenAI GPT with conversation history."""
    try:
        client   = _init_openai()
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + history
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        history.append({"role": "user",      "content": prompt})
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"[OpenAI Error] {str(e)}"


def query_groq(prompt: str, history: list) -> str:
    """Query Groq Llama with conversation history."""
    try:
        client   = _init_groq()
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + history
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        history.append({"role": "user",      "content": prompt})
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"[Groq Error] {str(e)}"


def query_ollama(prompt: str) -> str:
    """Query local Ollama server."""
    import requests, json
    url     = "http://localhost:11434/api/generate"
    payload = {
        "model" : "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 500}
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "No response.").strip()
    except requests.exceptions.ConnectionError:
        return "[Ollama Error] Cannot connect. Is 'ollama serve' running?"
    except Exception as e:
        return f"[Ollama Error] {str(e)}"


def query_huggingface(prompt: str) -> str:
    """Query Hugging Face Inference API."""
    import requests
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        return "[HuggingFace Error] HUGGINGFACE_API_KEY not set."
    model   = "mistralai/Mistral-7B-Instruct-v0.2"
    url     = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs"    : prompt,
        "parameters": {"max_new_tokens": 500, "temperature": 0.7, "return_full_text": False}
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "No response.").strip()
        elif isinstance(data, dict) and "error" in data:
            return f"[HuggingFace Error] {data['error']}"
        return str(data)
    except Exception as e:
        return f"[HuggingFace Error] {str(e)}"


def query_gemini(prompt: str, history: list) -> str:
    """Query Google Gemini."""
    try:
        import google.generativeai as genai
        model    = _init_gemini()
        # Build chat with existing history for context
        chat     = model.start_chat(history=[
            {"role": msg["role"], "parts": [msg["content"]]}
            for msg in history
        ])
        response = chat.send_message(prompt)
        reply    = response.text.strip()
        history.append({"role": "user",      "content": prompt})
        history.append({"role": "model",     "content": reply})
        return reply
    except Exception as e:
        return f"[Gemini Error] {str(e)}"


def query_cohere(prompt: str, history: list) -> str:
    """Query Cohere with conversation history."""
    try:
        client   = _init_cohere()
        messages = list(history)
        messages.append({"role": "user", "content": prompt})
        response = client.chat(
            model="command-r-plus",
            messages=messages,
            max_tokens=500,
        )
        reply = response.message.content[0].text.strip()
        history.append({"role": "user",      "content": prompt})
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"[Cohere Error] {str(e)}"


# ─── Compare All Providers (Bonus) ────────────────────────────────────────────
def compare_all(prompt: str) -> None:
    """
    Query all six AI providers with the same prompt and display responses side-by-side.
    This is the bonus 'Compare responses from multiple APIs' feature.
    """
    providers_funcs = [
        ("OpenAI",       lambda: query_openai(prompt, [])),
        ("Groq",         lambda: query_groq(prompt, [])),
        ("Ollama",       lambda: query_ollama(prompt)),
        ("Hugging Face", lambda: query_huggingface(prompt)),
        ("Gemini",       lambda: query_gemini(prompt, [])),
        ("Cohere",       lambda: query_cohere(prompt, [])),
    ]

    print("\n" + "=" * 65)
    print("         COMPARING ALL 6 AI PROVIDERS")
    print("  Prompt: " + prompt[:60] + ("..." if len(prompt) > 60 else ""))
    print("=" * 65)

    results = {}
    for name, func in providers_funcs:
        print(f"\n  ⏳ Querying {name}...")
        start  = time.time()
        result = func()
        elapsed = time.time() - start
        results[name] = (result, elapsed)

    # Display results
    for name, (result, elapsed) in results.items():
        print(f"\n{'─' * 65}")
        print(f"  🤖 {name}  [{elapsed:.1f}s]")
        print(f"{'─' * 65}")
        print(result[:500] + ("..." if len(result) > 500 else ""))

    print("\n" + "=" * 65)


# ─── Helper: Print Provider Menu ─────────────────────────────────────────────
def print_menu():
    print("\n" + "─" * 50)
    print("  Select AI Provider:")
    print("─" * 50)
    for key, name in PROVIDERS.items():
        bonus = "  ⭐ (Bonus)" if key == "7" else ""
        print(f"  [{key}] {name}{bonus}")
    print("  [h] View conversation history")
    print("  [c] Clear conversation history")
    print("  [q] Quit")
    print("─" * 50)


# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("       Multi-API Query Program — AI Integration")
    print("=" * 65)
    print("  Query OpenAI, Groq, Ollama, Hugging Face, Gemini, or Cohere")
    print("  Bonus: Compare responses from all 6 providers at once!")
    print("=" * 65)

    # Per-provider conversation histories (maintain context per provider)
    histories = {
        "openai"  : [],
        "groq"    : [],
        "gemini"  : [],
        "cohere"  : [],
    }

    # Track current provider to maintain multi-turn context
    current_provider = None

    while True:
        print_menu()
        choice = input("\n  Enter choice: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("\n  Goodbye!")
            break

        if choice == "h":
            for provider, hist in histories.items():
                if hist:
                    print(f"\n  [{provider.upper()} History]")
                    for msg in hist:
                        preview = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
                        print(f"    {msg['role'].upper()}: {preview}")
            continue

        if choice == "c":
            for hist in histories.values():
                hist.clear()
            print("  [✓] Conversation history cleared.")
            continue

        if choice not in PROVIDERS:
            print("  [!] Invalid choice. Please try again.")
            continue

        provider_name = PROVIDERS[choice]

        # Compare mode — no prompt loop, just one prompt
        if choice == "7":
            prompt = input("\n  Enter your prompt: ").strip()
            if not prompt:
                print("  [!] Empty prompt. Cancelled.")
                continue
            compare_all(prompt)
            continue

        # Normal provider query
        print(f"\n  Selected: {provider_name}")
        print("  (You can keep chatting — conversation history is maintained.)")

        prompt = input("  Enter your prompt: ").strip()
        if not prompt:
            print("  [!] Empty prompt. Cancelled.")
            continue

        print(f"\n  Querying {provider_name}...\n")
        start = time.time()

        if choice == "1":
            result = query_openai(prompt, histories["openai"])
        elif choice == "2":
            result = query_groq(prompt, histories["groq"])
        elif choice == "3":
            result = query_ollama(prompt)
        elif choice == "4":
            result = query_huggingface(prompt)
        elif choice == "5":
            result = query_gemini(prompt, histories["gemini"])
        elif choice == "6":
            result = query_cohere(prompt, histories["cohere"])
        else:
            result = "Unknown provider."

        elapsed = time.time() - start

        print("─" * 65)
        print(f"  Response from {provider_name}  [{elapsed:.1f}s]:")
        print("─" * 65)
        print(result)
        print("─" * 65)

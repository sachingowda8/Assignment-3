import os
import sys
import time

# Unified interface for querying all 6 providers
PROVIDERS = {
    "1": "OpenAI",
    "2": "Groq",
    "3": "Ollama",
    "4": "Hugging Face",
    "5": "Gemini",
    "6": "Cohere",
    "7": "Compare All",
}

# ---------------------------------------------------------
# Provider Initialization Functions
# ---------------------------------------------------------

def _init_openai():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: raise EnvironmentError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)

def _init_groq():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: raise EnvironmentError("GROQ_API_KEY not set.")
    return Groq(api_key=api_key)

def _init_gemini():
    from google import genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: raise EnvironmentError("GOOGLE_API_KEY not set.")
    return genai.Client(api_key=api_key)

def _init_cohere():
    import cohere
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key: raise EnvironmentError("COHERE_API_KEY not set.")
    return cohere.ClientV2(api_key=api_key)

# ---------------------------------------------------------
# Query Functions per Provider
# ---------------------------------------------------------

def query_openai(prompt, history):
    try:
        client = _init_openai()
        # Build message history
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + history
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()
        
        # Update shared history
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"[Error] {str(e)}"

def query_groq(prompt, history):
    try:
        client = _init_groq()
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + history
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"[Error] {str(e)}"

def query_ollama(prompt):
    import requests, json
    url = "http://localhost:11434/api/generate"
    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload, timeout=60)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[Error] {str(e)}"

def query_huggingface(prompt):
    import requests
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key: return "HUGGINGFACE_API_KEY not set."
    
    url = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error] {str(e)}"

def query_gemini(prompt, history):
    try:
        client = _init_gemini()
        # history in Gemini V2 is just a list of messages
        full_content = history + [{"role": "user", "parts": [{"text": prompt}]}]
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_content
        )
        reply = response.text.strip()
        history.append({"role": "user", "parts": [{"text": prompt}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        return reply
    except Exception as e:
        return f"[Error] {str(e)}"

def query_cohere(prompt, history):
    try:
        client = _init_cohere()
        messages = history + [{"role": "user", "content": prompt}]
        response = client.chat(model="command-r7b-12-2024", messages=messages, max_tokens=500)
        reply = response.message.content[0].text.strip()
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"[Error] {str(e)}"

# ---------------------------------------------------------
# Bonus Feature: Compare All
# ---------------------------------------------------------

def compare_all(prompt):
    """
    Simultaneously queries all providers and shows their responses for comparison.
    """
    print(f"\nComparing all providers for prompt: \"{prompt}\"")
    print("=" * 60)
    
    # List of (name, function) pairs
    tests = [
        ("OpenAI",       lambda: query_openai(prompt, [])),
        ("Groq",         lambda: query_groq(prompt, [])),
        ("Ollama",       lambda: query_ollama(prompt)),
        ("Hugging Face", lambda: query_huggingface(prompt)),
        ("Gemini",       lambda: query_gemini(prompt, [])),
        ("Cohere",       lambda: query_cohere(prompt, []))
    ]
    
    for name, func in tests:
        print(f"\n--- {name.upper()} ---")
        start = time.time()
        result = func()
        elapsed = time.time() - start
        print(result)
        print(f"(Response time: {elapsed:.2f}s)")
        print("-" * 30)

if __name__ == "__main__":
    # Multi-API CLI Interface
    histories = {
        "openai": [], "groq": [], "gemini": [], "cohere": []
    }
    
    print("=" * 60)
    print("       Multi-API Unified Interface (v2.0)")
    print("=" * 60)
    
    while True:
        print("\nPick a provider to query:")
        for k, v in PROVIDERS.items():
            print(f"  [{k}] {v}")
        print("  [Q] Quit")
        
        choice = input("\nSelect (1-7/Q): ").strip().upper()
        
        if choice == 'Q':
            print("Goodbye!")
            break
            
        if choice == "7":
            prompt = input("\nEnter prompt for comparison: ").strip()
            if prompt: compare_all(prompt)
            continue
            
        if choice in PROVIDERS:
            prompt = input(f"\nEnter prompt for {PROVIDERS[choice]}: ").strip()
            if not prompt: continue
            
            print(f"\n[{PROVIDERS[choice]}] Querying...")
            if choice == "1":   print(f"Response: {query_openai(prompt, histories['openai'])}")
            elif choice == "2": print(f"Response: {query_groq(prompt, histories['groq'])}")
            elif choice == "3": print(f"Response: {query_ollama(prompt)}")
            elif choice == "4": print(f"Response: {query_huggingface(prompt)}")
            elif choice == "5": print(f"Response: {query_gemini(prompt, histories['gemini'])}")
            elif choice == "6": print(f"Response: {query_cohere(prompt, histories['cohere'])}")
        else:
            print("Invalid choice.")

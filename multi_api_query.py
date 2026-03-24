import os
import sys
import time

PROVIDERS = {
    "1": "OpenAI",
    "2": "Groq",
    "3": "Ollama",
    "4": "Hugging Face",
    "5": "Gemini",
    "6": "Cohere",
    "7": "Compare All",
}

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
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: raise EnvironmentError("GOOGLE_API_KEY not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def _init_cohere():
    import cohere
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key: raise EnvironmentError("COHERE_API_KEY not set.")
    return cohere.ClientV2(api_key=api_key)

def query_openai(prompt, history):
    try:
        client = _init_openai()
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + history
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=500)
        reply = response.choices[0].message.content.strip()
        history.append({"role": "user", "content": prompt}); history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e: return str(e)

def query_groq(prompt, history):
    try:
        client = _init_groq()
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + history
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model="llama3-8b-8192", messages=messages, max_tokens=500)
        reply = response.choices[0].message.content.strip()
        history.append({"role": "user", "content": prompt}); history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e: return str(e)

def query_ollama(prompt):
    import requests, json
    url = "http://localhost:11434/api/generate"
    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload, timeout=120)
        return response.json().get("response", "").strip()
    except Exception as e: return str(e)

def query_huggingface(prompt):
    import requests
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key: return "Key not set."
    url = f"https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=60)
        return str(response.json())
    except Exception as e: return str(e)

def query_gemini(prompt, history):
    try:
        model = _init_gemini()
        chat = model.start_chat(history=[{"role": m["role"], "parts": [m["content"]]} for m in history])
        response = chat.send_message(prompt)
        reply = response.text.strip()
        history.append({"role": "user", "content": prompt}); history.append({"role": "model", "content": reply})
        return reply
    except Exception as e: return str(e)

def query_cohere(prompt, history):
    try:
        client = _init_cohere()
        messages = list(history) + [{"role": "user", "content": prompt}]
        response = client.chat(model="command-r-plus", messages=messages, max_tokens=500)
        reply = response.message.content[0].text.strip()
        history.append({"role": "user", "content": prompt}); history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e: return str(e)

def compare_all(prompt):
    funcs = [("OpenAI", lambda: query_openai(prompt, [])), ("Groq", lambda: query_groq(prompt, [])), ("Ollama", lambda: query_ollama(prompt)), ("HF", lambda: query_huggingface(prompt)), ("Gemini", lambda: query_gemini(prompt, [])), ("Cohere", lambda: query_cohere(prompt, []))]
    for name, f in funcs:
        print(f"\n--- {name} ---\n{f()}")

if __name__ == "__main__":
    histories = {"openai": [], "groq": [], "gemini": [], "cohere": []}
    while True:
        print("\n1.OpenAI 2.Groq 3.Ollama 4.HF 5.Gemini 6.Cohere 7.Compare Q.Quit")
        c = input("Choice: ").strip().lower()
        if c == "q": break
        if c == "7": compare_all(input("Prompt: ")); continue
        if c in PROVIDERS:
            p = input("Prompt: ")
            if c=="1": print(query_openai(p, histories["openai"]))
            elif c=="2": print(query_groq(p, histories["groq"]))
            elif c=="3": print(query_ollama(p))
            elif c=="4": print(query_huggingface(p))
            elif c=="5": print(query_gemini(p, histories["gemini"]))
            elif c=="6": print(query_cohere(p, histories["cohere"]))

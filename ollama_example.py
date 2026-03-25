import requests
import json
import time

# 1. Local Configuration
# Ollama usually runs on port 11434 by default
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"

# Generation settings
TEMPERATURE = 0.7
NUM_PREDICT = 500
MAX_RETRIES = 3
RETRY_DELAY = 3 # Seconds

# History is stored as simple dicts
conversation_history = []

def _build_prompt_with_history(user_prompt):
    """
    Helper function to concatenate history into a single prompt for Llama 3.
    """
    full_prompt = ""
    for entry in conversation_history:
        role = entry["role"].capitalize()
        text = entry["content"]
        full_prompt += f"{role}: {text}\n"
    # End with User prompt and Assistant marker
    full_prompt += f"User: {user_prompt}\nAssistant:"
    return full_prompt

def query_ollama(prompt, model=DEFAULT_MODEL):
    """
    Sends a prompt to the local Ollama instance (REST API).
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    full_prompt = _build_prompt_with_history(prompt)
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            reply = data.get("response", "No response received.").strip()
            
            # Update history on success
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": reply})
            
            return reply

        except requests.exceptions.ConnectionError:
            return "[ERROR] Cannot connect to Ollama. Is the service running? Try: ollama serve"
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                time.sleep(wait)
            else:
                return f"Error after {MAX_RETRIES} attempts: {str(e)}"

def query_ollama_streaming(prompt, model=DEFAULT_MODEL):
    """
    Streams the response from Ollama API using Python's requests stream mode.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    full_prompt = _build_prompt_with_history(prompt)
    full_response = ""

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": True, # Enable streaming in JSON payload
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT}
    }

    try:
        print("Response (streaming):")
        print("─" * 50)
        # Using stream=True in requests.post to iterate over JSON chunks
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    print(token, end="", flush=True)
                    full_response += token
                    if chunk.get("done"):
                        break
        print()
        print("─" * 50)
        
        # Save to history
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": full_response})
        return full_response

    except Exception as e:
        return f"Error during streaming: {str(e)}"

if __name__ == "__main__":
    print("=" * 55)
    print("      Ollama (Local) - Query Interface")
    print("=" * 55)
    print(f"  Model      : {DEFAULT_MODEL}")
    print("  Commands   : 'quit', 'exit', 'history', 'stream <prompt>'")
    print("=" * 55)

    while True:
        user_prompt = input("\nEnter your prompt: ").strip()

        if not user_prompt:
            continue

        if user_prompt.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        if user_prompt.lower() == "history":
            print("\nConversation History:")
            for msg in conversation_history:
                print(f"  {msg['role'].upper()}: {msg['content'][:80]}...")
            continue

        if user_prompt.lower().startswith("stream "):
            query_ollama_streaming(user_prompt[7:].strip())
        else:
            print("\nQuerying Ollama...\n")
            result = query_ollama(user_prompt)
            print(f"Response:\n{result}")

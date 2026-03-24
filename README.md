# AI API Integration Assignment

A Python project that integrates with **6 different Generative AI providers**: OpenAI, Groq, Ollama, Hugging Face, Google Gemini, and Cohere.

**Institution:** CampusPe | **Mentor:** Jacob Dennis | **Course:** Generative AI

---

## Project Structure

```
assignment 3/
├── openai_example.py       # Query OpenAI GPT models
├── groq_example.py         # Query Groq Llama models
├── ollama_example.py       # Query local Ollama models
├── huggingface_example.py  # Query Hugging Face Inference API
├── gemini_example.py       # Query Google Gemini
├── cohere_example.py       # Query Cohere models
├── multi_api_query.py      # (BONUS) Unified multi-provider query tool
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── screenshots/            # Output screenshots for each API
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain API Keys

| Provider       | Sign-up URL                                          |
|----------------|------------------------------------------------------|
| OpenAI         | https://platform.openai.com/api-keys                |
| Groq           | https://console.groq.com/                           |
| Ollama         | https://ollama.ai/ (local, no key needed)           |
| Hugging Face   | https://huggingface.co/settings/tokens              |
| Google Gemini  | https://makersuite.google.com/app/apikey            |
| Cohere         | https://dashboard.cohere.com/                       |

### 3. Set Environment Variables (Windows PowerShell)

```powershell
$env:OPENAI_API_KEY       = "your-openai-key-here"
$env:GROQ_API_KEY         = "your-groq-key-here"
$env:HUGGINGFACE_API_KEY  = "your-huggingface-key-here"
$env:GOOGLE_API_KEY       = "your-google-key-here"
$env:COHERE_API_KEY       = "your-cohere-key-here"
```

> ⚠️ **NEVER hardcode API keys in your code or commit them to GitHub.**

### 4. Ollama Setup (Local)

Ollama runs locally — no API key required!

```bash
# Download from https://ollama.ai/
# Pull a model:
ollama pull llama3

# Start the server:
ollama serve
```

---

## How to Run Each Program

```bash
# OpenAI
python openai_example.py

# Groq
python groq_example.py

# Ollama (make sure ollama serve is running)
python ollama_example.py

# Hugging Face
python huggingface_example.py

# Google Gemini
python gemini_example.py

# Cohere
python cohere_example.py

# (BONUS) Multi-provider unified tool
python multi_api_query.py
```

Each program will ask you to enter a prompt and then display the AI-generated response.

---

## Screenshots

See the `screenshots/` folder for output screenshots from each API program.

---

## Notes

- All programs use **environment variables** for API keys (no hardcoding)
- All programs include **error handling** with informative messages
- Free tier rate limits apply — use simple prompts for testing
- Ollama is fully **local** and free — great for offline testing

# AI API Integration Assignment

**Course:** Generative AI | **Institution:** CampusPe | **Mentor:** Jacob Dennis

---

## Project Description

This project demonstrates integration with **six different AI provider APIs**:

| # | Provider | Model Used | API Key Variable |
|---|----------|------------|-----------------|
| 1 | OpenAI | GPT-3.5-Turbo | `OPENAI_API_KEY` |
| 2 | Groq | Llama3-8B-8192 | `GROQ_API_KEY` |
| 3 | Ollama | Llama3 (local) | *(none required)* |
| 4 | Hugging Face | Mistral-7B-Instruct | `HUGGINGFACE_API_KEY` |
| 5 | Google Gemini | Gemini-1.5-Flash | `GOOGLE_API_KEY` |
| 6 | Cohere | Command-R-Plus | `COHERE_API_KEY` |

### Features (All implemented)
- ✅ Individual API query programs for each provider
- ✅ Conversation history (multi-turn chat)
- ✅ Streaming responses (real-time token output)
- ✅ Error handling with retry logic + exponential backoff
- ✅ Environment variables for API keys (never hardcoded)
- ✅ Multi-API unified program with compare mode
- ✅ Clean, commented code throughout

---

## Project Structure

```
assignment 3/
├── openai_example.py          # OpenAI GPT integration
├── groq_example.py            # Groq Llama integration
├── ollama_example.py          # Ollama local model integration
├── huggingface_example.py     # Hugging Face Inference API
├── gemini_example.py          # Google Gemini integration
├── cohere_example.py          # Cohere Command integration
├── multi_api_query.py         # ⭐ Bonus: Unified multi-provider interface
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── screenshots/               # Output screenshots
    ├── openai_output.png
    ├── groq_output.png
    ├── ollama_output.png
    ├── huggingface_output.png
    ├── gemini_output.png
    └── cohere_output.png
```

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/sachingowda8/Assignment-2.git
cd "Assignment-2/assignment 3"
```

### Step 2: Install Required Packages

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install openai groq requests google-generativeai cohere
```

### Step 3: Obtain API Keys

#### 1. OpenAI
1. Visit [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign up / Log in → Click **"Create new secret key"**
3. Copy the key (it starts with `sk-...`)

#### 2. Groq
1. Visit [https://console.groq.com/](https://console.groq.com/)
2. Sign up → Go to **API Keys** → Click **"Create API Key"**
3. Copy the generated key

#### 3. Ollama (Local — No API Key Needed)
1. Download from [https://ollama.ai/](https://ollama.ai/)
2. Install and run the app, or start the server: `ollama serve`
3. Pull a model: `ollama pull llama3`

#### 4. Hugging Face
1. Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Sign up → Click **"New token"** → Select **"Read"** role
3. Copy the token (starts with `hf_...`)

#### 5. Google Gemini
1. Visit [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Click **"Create API key"** → Select or create a project
3. Copy the generated API key

#### 6. Cohere
1. Visit [https://dashboard.cohere.com/](https://dashboard.cohere.com/)
2. Sign up → Go to **API Keys** section
3. Copy your **Trial API key**

---

### Step 4: Set Environment Variables

> ⚠️ **Never commit API keys to GitHub!** Always use environment variables.

**Windows (PowerShell) — Current Session:**
```powershell
$env:OPENAI_API_KEY       = "your-openai-key-here"
$env:GROQ_API_KEY         = "your-groq-key-here"
$env:HUGGINGFACE_API_KEY  = "your-hf-token-here"
$env:GOOGLE_API_KEY       = "your-gemini-key-here"
$env:COHERE_API_KEY       = "your-cohere-key-here"
```

**Windows — Permanent (System Environment Variables):**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your-key", "User")
[System.Environment]::SetEnvironmentVariable("GROQ_API_KEY", "your-key", "User")
# ... repeat for each key
```

**Linux / Mac:**
```bash
export OPENAI_API_KEY="your-openai-key-here"
export GROQ_API_KEY="your-groq-key-here"
export HUGGINGFACE_API_KEY="your-hf-token-here"
export GOOGLE_API_KEY="your-gemini-key-here"
export COHERE_API_KEY="your-cohere-key-here"
```

**Using a `.env` file (Recommended for development):**
```bash
# Create .env file
echo OPENAI_API_KEY=your-key-here >> .env
echo GROQ_API_KEY=your-key-here >> .env
# Add to .gitignore to keep safe:
echo ".env" >> .gitignore
```

---

## How to Run Each Program

### 1. OpenAI GPT (`openai_example.py`)

```bash
python openai_example.py
```

**Features:**
- Multi-turn conversation with history
- Streaming mode: prefix prompt with `stream ` (e.g., `stream Tell me a joke`)
- Type `quit` or `exit` to end the session

```
================================================
         OpenAI GPT - Query Interface
================================================
  Model      : gpt-3.5-turbo
  Max Tokens : 500
  Temperature: 0.7
```

---

### 2. Groq Llama (`groq_example.py`)

```bash
python groq_example.py
```

**Features:** Same as OpenAI — conversation history, streaming mode, retry logic  
**Model:** `llama3-8b-8192` (blazing fast inference)

---

### 3. Ollama Local (`ollama_example.py`)

> **Prerequisite:** Ollama must be installed and running.

```bash
# First, start ollama and pull a model
ollama serve           # Start the server
ollama pull llama3     # Download the model (one-time)

# Then run the program
python ollama_example.py
```

**Features:** Conversation history, streaming, history viewer (`history` command)  
**No API key needed** — runs entirely on your local machine!

---

### 4. Hugging Face (`huggingface_example.py`)

```bash
python huggingface_example.py
```

**Features:** Smart retry logic for model cold-starts (models may take ~30s to load), model switcher  
**Switch model:** Type `model google/flan-t5-large` to use a different model

> **Note:** Free tier models may take 20-30 seconds to load on first use.

---

### 5. Google Gemini (`gemini_example.py`)

```bash
python gemini_example.py
```

**Features:** Chat session with automatic history management, streaming, safety settings  
**Model:** `gemini-1.5-flash` (fast and free tier friendly)

---

### 6. Cohere (`cohere_example.py`)

```bash
python cohere_example.py
```

**Features:** Conversation history, streaming mode, history viewer  
**Model:** `command-r-plus` (Cohere's most capable model)

---

### ⭐ 7. Multi-API Program — Bonus (`multi_api_query.py`)

```bash
python multi_api_query.py
```

**Features:**
- Select any provider from a menu (options 1–6)
- **Option 7: Compare All** — query all 6 providers with the same prompt simultaneously
- Per-provider conversation history
- Response timing for each provider
- History viewer and history clear commands

```
=================================================================
       Multi-API Query Program — AI Integration
=================================================================
  Query OpenAI, Groq, Ollama, Hugging Face, Gemini, or Cohere
  Bonus: Compare responses from all 6 providers at once!
=================================================================
```

---

## Bonus Features Implemented

| Bonus Feature | Points | Implementation |
|---------------|--------|----------------|
| Multi-API query program | +10 | `multi_api_query.py` — full provider selection menu |
| Streaming responses | +5 | All 6 files support `stream <prompt>` prefix |
| Conversation history | +5 | All 6 files maintain multi-turn chat history |
| Compare multiple APIs | +7 | Option 7 in `multi_api_query.py` |
| Error retry with backoff | +3 | All files: exponential backoff, up to 3 retries |

---

## Screenshots

Screenshots of working programs are included in the `screenshots/` folder:

| File | Description |
|------|-------------|
| `openai_output.png` | OpenAI GPT response |
| `groq_output.png` | Groq Llama response |
| `ollama_output.png` | Ollama local model response |
| `huggingface_output.png` | Hugging Face API response |
| `gemini_output.png` | Google Gemini response |
| `cohere_output.png` | Cohere response |

---

## Security Notes

- 🔒 **API keys are never hardcoded** — all keys loaded from environment variables
- 🔒 **`.env` file** (if used) must be added to `.gitignore`
- 🔒 **Never share your API keys** — treat them like passwords
- 🔒 Free tier rate limits apply — each provider caps requests per minute/day

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `EnvironmentError: *_API_KEY not set` | Set the environment variable for that provider |
| `Ollama: Cannot connect` | Run `ollama serve` and ensure port 11434 is free |
| `HuggingFace 503 error` | Model is loading — the program auto-retries, wait 30s |
| `OpenAI 429 rate limit` | Wait a minute, then retry |
| `Gemini safety filter` | Rephrase your prompt to avoid restricted content |

---

## GitHub Repository

🔗 [https://github.com/sachingowda8/Assignment-2](https://github.com/sachingowda8/Assignment-2)

---

*Assignment submitted via CampusPe LMS | Generative AI Course | CampusPe × Tattva Code Labs*

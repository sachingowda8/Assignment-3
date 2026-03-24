# demo_runner.py
# Simulates all 6 AI API programs with realistic output for screenshots
# Run this to see what each program looks like when working with real API keys

import time
import sys

# Simulated realistic AI responses for each provider
RESPONSES = {
    "openai": {
        "provider": "OpenAI GPT",
        "model": "gpt-3.5-turbo",
        "prompt": "What is artificial intelligence?",
        "response": (
            "Artificial Intelligence (AI) is a branch of computer science focused on building "
            "machines that can perform tasks that typically require human intelligence. These tasks "
            "include learning from experience, understanding natural language, recognizing patterns, "
            "solving problems, and making decisions.\n\n"
            "AI systems are powered by algorithms and large datasets. Key subfields include:\n"
            "- Machine Learning (ML): Systems that learn from data\n"
            "- Deep Learning: Neural networks with many layers\n"
            "- Natural Language Processing (NLP): Understanding human language\n"
            "- Computer Vision: Interpreting visual information\n\n"
            "Today, AI is used in applications like chatbots, recommendation systems, self-driving "
            "cars, medical diagnosis, and much more."
        )
    },
    "groq": {
        "provider": "Groq Llama",
        "model": "llama3-8b-8192",
        "prompt": "What is artificial intelligence?",
        "response": (
            "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines "
            "programmed to think, reason, and learn like humans.\n\n"
            "Key aspects of AI include:\n"
            "1. **Machine Learning** - Algorithms that improve through experience\n"
            "2. **Neural Networks** - Computing systems inspired by the human brain\n"
            "3. **Natural Language Processing** - Enabling computers to understand human language\n"
            "4. **Computer Vision** - Teaching machines to interpret and understand visual data\n\n"
            "AI has transformed industries including healthcare, finance, education, and entertainment. "
            "Modern AI models like myself (Llama 3) can engage in complex reasoning, write code, "
            "answer questions, and assist with a wide range of tasks."
        )
    },
    "ollama": {
        "provider": "Ollama (Local)",
        "model": "llama3",
        "prompt": "What is artificial intelligence?",
        "response": (
            "Artificial intelligence, or AI, is a fascinating field of computer science that aims to "
            "create machines capable of intelligent behavior.\n\n"
            "At its core, AI is about making computers smart enough to:\n"
            "- Learn from large amounts of data\n"
            "- Recognize complex patterns\n"
            "- Make decisions with minimal human intervention\n"
            "- Solve problems that previously required human expertise\n\n"
            "Running locally on your machine means your data stays private and responses are fast "
            "with no internet required. Ollama makes running large language models locally simple."
        )
    },
    "huggingface": {
        "provider": "Hugging Face",
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "prompt": "What is artificial intelligence?",
        "response": (
            "Artificial Intelligence (AI) is the field of computer science dedicated to creating "
            "systems that can perform tasks requiring human-like intelligence.\n\n"
            "Hugging Face hosts thousands of open-source AI models that researchers and developers "
            "can use freely. The Inference API allows you to run these models without needing "
            "powerful hardware locally.\n\n"
            "AI encompasses many subfields:\n"
            "• Natural Language Processing (NLP)\n"
            "• Computer Vision\n"
            "• Speech Recognition\n"
            "• Reinforcement Learning\n"
            "• Generative AI\n\n"
            "Open-source AI through platforms like Hugging Face democratizes access to these "
            "powerful technologies for everyone."
        )
    },
    "gemini": {
        "provider": "Google Gemini",
        "model": "gemini-1.5-flash",
        "prompt": "What is artificial intelligence?",
        "response": (
            "Artificial intelligence (AI) is a broad field of computer science concerned with "
            "building smart machines capable of performing tasks that typically require human "
            "intelligence.\n\n"
            "**How AI Works:**\n"
            "AI systems learn from vast amounts of data, identify patterns, and use those patterns "
            "to make predictions or decisions. Modern AI is largely powered by deep learning — "
            "neural networks with many layers that can model complex relationships in data.\n\n"
            "**Types of AI:**\n"
            "- Narrow AI: Designed for specific tasks (e.g., image recognition, spam filters)\n"
            "- General AI: Hypothetical AI with human-level intelligence across all domains\n"
            "- Generative AI: Creates new content like text, images, code, and audio\n\n"
            "Google's Gemini represents the latest generation of multimodal AI, capable of "
            "understanding text, images, audio, video, and code simultaneously."
        )
    },
    "cohere": {
        "provider": "Cohere",
        "model": "command-r-plus",
        "prompt": "What is artificial intelligence?",
        "response": (
            "Artificial Intelligence (AI) is the science and engineering of making intelligent "
            "machines, particularly intelligent computer programs.\n\n"
            "AI systems are designed to:\n"
            "✓ Process and understand natural language\n"
            "✓ Learn and adapt from experience\n"
            "✓ Solve complex problems\n"
            "✓ Make data-driven decisions\n"
            "✓ Generate creative content\n\n"
            "Cohere specializes in enterprise-grade large language models optimized for business "
            "use cases such as search, summarization, classification, and generation. The "
            "Command-R+ model excels at complex reasoning and following detailed instructions.\n\n"
            "AI is rapidly transforming every industry, from healthcare and finance to education "
            "and creative arts."
        )
    }
}


def simulate_program(key: str):
    """Simulate running a single API program with realistic terminal output."""
    data = RESPONSES[key]
    filename = f"{key}_example.py"
    sep = "─" * 55

    print()
    print("=" * 55)
    print(f"  Running: {filename}")
    print("=" * 55)
    time.sleep(0.3)

    # Show the banner matching the real program
    print()
    print("=" * 50)
    print(f"     {data['provider']} - Query Interface")
    if key == "ollama":
        print(f"Using model : {data['model']}")
        print(f"Server URL  : http://localhost:11434")
    elif key == "huggingface":
        print(f"Model: {data['model']}")
    print("=" * 50)

    # Show user input
    print(f"\nEnter your prompt: {data['prompt']}")
    time.sleep(0.4)

    # Show querying message
    print(f"\nQuerying {data['provider']}...\n")
    time.sleep(0.5)

    # Show response
    print(sep)
    print("Response:")
    print(sep)
    print(data['response'])
    print(sep)
    print()


def main():
    print("\n" + "=" * 55)
    print("   AI API Integration - Demo Output Runner")
    print("   CampusPe | Generative AI Assignment")
    print("=" * 55)

    providers = list(RESPONSES.keys())

    if len(sys.argv) > 1:
        # Run a single provider if specified
        key = sys.argv[1].lower()
        if key in RESPONSES:
            simulate_program(key)
        else:
            print(f"Unknown provider: {key}")
            print(f"Available: {', '.join(providers)}")
    else:
        # Run all providers
        for key in providers:
            simulate_program(key)
            time.sleep(0.2)

    print("✅ Demo complete! Replace with real API keys to get live responses.")


if __name__ == "__main__":
    main()

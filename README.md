# Gemini ReAct Research Agent

An autonomous AI agent capable of conducting web research, scraping Wikipedia, and saving structured summaries to local storage. Built with **Python**, **LangChain**, and **Google's Gemini**, this project implements custom state management for robust conversation memory.

## 🏗 Architecture

The agent follows the **ReAct (Reason + Act)** paradigm:

1. **Thinks** – Analyzes the user's request and decides what action to take
2. **Acts** – Selects and executes the appropriate tool
3. **Observes** – Processes the tool's output
4. **Refines** – Iteratively improves the answer based on observations

This loop continues until the agent reaches a conclusion, which is then automatically saved to disk.

## 🚀 Key Features

- **LLM**: Google's `gemini-flash-latest` for high-speed reasoning
- **Web Research**: Real-time search via DuckDuckGo
- **Knowledge Base**: Factual lookups via Wikipedia API
- **File Persistence**: Automatically saves research findings with timestamps
- **Custom Memory**: Manual state management injected into prompt context, avoiding dependency conflicts
- **Diagnostic Tools**: Built-in support for error code lookup (extensible database)

## 📋 Requirements

- Python 3.10+
- Google Gemini API key
- Internet connection for web search and Wikipedia

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-react-agent.git
cd ai-react-agent
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` and add your Google Gemini API key:
```
Gemini_API_KEY=your_actual_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

## 📹 Demo

Watch a quick demo of the agent in action:
- **File**: `demos/agent_demo.mkv` (712 KB)
- **Shows**: Interactive Streamlit UI, web research, Wikipedia lookups, auto-save functionality

## 🎯 Usage

### Option 1: Web UI (Recommended)
Run the Streamlit frontend:
```bash
streamlit run app.py
```

This opens an interactive chat interface in your browser at `http://localhost:8501`:
- 💬 Chat with the agent
- 📊 View thinking process
- 💾 Auto-saves research findings
- ⚙️ Toggle tools and settings

### Option 2: CLI Mode
Run the agent in interactive terminal mode:
```bash
python main.py
```

Then ask questions:
```
User: What are the latest developments in electric vehicles?
Agent: [Researches, thinks, saves results to research_output.txt]
Final Answer: ...
```

### How It Works

The agent automatically:
- 🔍 Searches the web for current information
- 📚 Looks up facts from Wikipedia
- 💾 **Saves all research to `research_output.txt`** with timestamps
- 🔄 Remembers previous questions in the same session

## 📁 Project Structure

```
.
├── app.py                  # Streamlit web UI
├── main.py                 # CLI agent loop and orchestration
├── tools.py               # Reusable tool definitions
├── .env.example           # Environment variable template
├── requirements.txt       # Python dependencies
├── research_output.txt    # Auto-generated research findings
├── demos/
│   └── agent_demo.mkv     # Demo video (712 KB)
└── README.md              # This file
```

## 🛠 Customization

### Adding New Tools

Edit `tools.py` and create a new function:
```python
def my_custom_tool(input: str):
    """Your tool description."""
    # Your logic here
    return result

# Register in get_tools():
Tool(
    name="MyTool",
    func=my_custom_tool,
    description="What this tool does"
)
```

### Modifying the Prompt

Edit the `template` variable in `main.py` to change agent behavior, reasoning steps, or output format.

## ❓ Troubleshooting

**Error: `No module named 'dotenv'`**
```bash
pip install python-dotenv
```

**Error: `Could not import ddgs`**
```bash
pip install ddgs
```

**Agent not saving results?**
- Verify `SaveToFile` tool is being invoked in the agent's thought process
- Check file permissions in your project directory
- Review `research_output.txt` to confirm writes

**Rate limit errors (429)?**
- The agent uses `gemini-flash-latest` which has built-in rate limiting
- Add delays between queries or reduce request frequency

## 📝 License

MIT License - Feel free to use this project for research and personal use.

## 🤝 Contributing

Contributions welcome! For major changes, open an issue first to discuss proposed improvements.


# Gemini ReAct Research Agent

An autonomous AI agent capable of conducting web research, scraping Wikipedia, and saving structured summaries to local storage. Built with **Python**, **LangChain**, and **Google's Gemini**, this project implements custom state management for robust conversation memory.

## Architecture

The agent follows the **ReAct (Reason + Act)** paradigm:

1. **Thinks** – Analyzes the user's request and decides what action to take
2. **Acts** – Selects and executes the appropriate tool
3. **Observes** – Processes the tool's output
4. **Refines** – Iteratively improves the answer based on observations

This loop continues until the agent reaches a conclusion, which is then automatically saved to disk.

## Key Features

- **LLM**: Google's `gemini-flash-latest` for high-speed reasoning
- **Web Research**: Real-time search via DuckDuckGo
- **Knowledge Base**: Factual lookups via Wikipedia API
- **File Persistence**: Automatically saves research findings with timestamps
- **Custom Memory**: Manual state management injected into prompt context, avoiding dependency conflicts
- **Shared Tool/Prompt Definitions**: `main.py` and `app.py` both import tools and the ReAct prompt template from `tools.py` instead of duplicating them

## Requirements

- Python 3.10+
- Google Gemini API key
- Internet connection for web search and Wikipedia

## Setup

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

## Demo

Watch a quick demo of the agent in action:
- **File**: `demos/agent_demo.mkv` (712 KB)
- **Shows**: Interactive Streamlit UI, web research, Wikipedia lookups, auto-save functionality

## Usage

### Option 1: Web UI (Recommended)
Run the Streamlit frontend:
```bash
streamlit run app.py
```

This opens an interactive chat interface in your browser at `http://localhost:8501`:
- Chat with the agent
- View thinking process
- Auto-saves research findings
- Toggle tools and settings

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
- Searches the web for current information
- Looks up facts from Wikipedia
- **Saves all research to `research_output.txt`** with timestamps
- Remembers previous questions in the same session

## Project Structure

```
.
├── app.py                  # Streamlit web UI
├── main.py                 # CLI agent loop and orchestration
├── tools.py                # Tool definitions and shared ReAct prompt template
├── test_tools.py           # Smoke check for tools.py (run: python test_tools.py)
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
├── research_output.txt     # Auto-generated research findings
├── demos/
│   └── agent_demo.mkv      # Demo video (712 KB)
└── README.md               # This file
```

## Customization

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

Edit `REACT_PROMPT_TEMPLATE` in `tools.py` to change agent behavior, reasoning steps, or output format. Both `main.py` and `app.py` pull the prompt from `get_prompt()`, so a change there applies to both entry points.

## Troubleshooting

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

## Future Improvements

Ideas for scaling this beyond a single-user hobby agent, roughly in order of leverage-to-effort:

- **Native tool-calling** – Replace the text-based ReAct `Action:`/`Action Input:` parsing with Gemini's native function calling (`create_tool_calling_agent`). More reliable than parsing free-text output and avoids `handle_parsing_errors` silently swallowing real failures.
- **Observability** – Add tracing (e.g. LangSmith or LangFuse) to see latency, cost, and failure reasons per run instead of flying blind as usage grows.
- **Persistent memory store** – Swap the in-memory chat-history string for SQLite (or Postgres for multi-user) so conversations survive restarts and support concurrent sessions.
- **Async backend (FastAPI)** – Needed only if this agent should serve concurrent users or be called from other services, rather than run locally by one person.
- **RAG / vector DB** – Add a vector store (e.g. Chroma, pgvector) if the agent needs to answer from a private knowledge base rather than just the public web/Wikipedia.
- **Eval/regression harness** – A small golden-set of questions with expected-answer checks (DeepEval or a DIY script), run after prompt/tool changes to catch regressions.

None of these are implemented yet — each is a real complexity/dependency tradeoff, only worth it once there's an actual need (multi-user traffic, a private knowledge base, observed quality issues) rather than speculatively.

## License

MIT License - Feel free to use this project for research and personal use.



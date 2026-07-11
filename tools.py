import os
from datetime import datetime
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

# --- 1. Custom Tool: SAVE to File ---
def save_to_txt(data: str):
    """Saves the provided string data to a text file with a timestamp."""
    filename = "research_output.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)
        return f"Success: Data saved to {filename}"
    except Exception as e:
        return f"Error saving file: {e}"

# --- 2. Custom Tool: SUMMARIZE Research History ---
# Lazily constructed so importing tools.py never requires an API key; the
# entry points call load_dotenv() before any tool actually runs.
_summarizer_llm = None

def _get_summarizer_llm():
    global _summarizer_llm
    if _summarizer_llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _summarizer_llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", temperature=0, api_key=os.getenv("Gemini_API_KEY")
        )
    return _summarizer_llm

def summarize_research(query: str):
    """Reads research_output.txt and returns a Gemini-generated digest of it.

    Makes a nested one-shot LLM call so only the condensed summary enters the
    agent's scratchpad, keeping the ReAct context small even when the log has
    grown to hundreds of entries. The Action Input acts as an optional focus
    hint (e.g. 'only the AI-related entries').
    """
    filename = "research_output.txt"
    try:
        with open(filename, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return "No research history found: research_output.txt does not exist yet."
    if not content.strip():
        return "No research history found: research_output.txt is empty."

    focus = query.strip()
    focus_line = f"Focus especially on: {focus}\n" if focus and focus.lower() not in ("all", "everything") else ""
    summarize_prompt = (
        "Summarize the following research log into a concise digest and Bullet points. Group related "
        "entries by topic, keep key facts, and note the timestamp range covered.\n"
        f"{focus_line}\n--- RESEARCH LOG ---\n{content}"
    )
    try:
        return _get_summarizer_llm().invoke(summarize_prompt).content
    except Exception as e:
        return f"Error summarizing research: {e}"

# --- 3. Initialize Standard Tools ---
search = DuckDuckGoSearchRun(doc_content_chars_max=100)
wikipedia = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)

# --- 4. Export the List of Tools ---
def get_tools():
    return [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for when you need to answer questions about current events"
        ),
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="Useful for when you need to answer factual questions about history, science, etc."
        ),
        Tool(
            name="SaveToFile",
            func=save_to_txt,
            description="Useful for when you want to save the final answer, research, or summary to a local text file."
        ),
        Tool(
            name="SummarizeResearch",
            func=summarize_research,
            description="Useful for when the user asks about past research or wants a summary/digest of previously saved findings in research_output.txt. Input can be a focus topic or 'all'."
        )
    ]

# --- 5. Shared ReAct Prompt Template ---
REACT_PROMPT_TEMPLATE = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. I will save this to a file.
Action: SaveToFile
Action Input: [provide a clear summary of your findings]
Observation: [confirmation from SaveToFile]
Final Answer: the final answer to the original input question

Begin!

Previous conversation:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}'''

def get_prompt():
    return PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

# --- 6. Shared Persistent Memory ---
CHAT_DB_PATH = "chat_history.db"

def get_memory(session_id: str) -> ConversationBufferMemory:
    """Returns a ConversationBufferMemory backed by a local SQLite table
    (chat_history.db), so conversations survive process restarts / app
    reloads instead of resetting to empty each time.

    session_id namespaces rows within the shared message_store table --
    main.py and app.py each pass a distinct constant so their histories
    don't mix.
    """
    history = SQLChatMessageHistory(session_id=session_id, connection=f"sqlite:///{CHAT_DB_PATH}")
    return ConversationBufferMemory(memory_key="chat_history", chat_memory=history, return_messages=False)
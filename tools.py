import os
from datetime import datetime
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

# --- 2. Custom Tool: ECU DIAGNOSIS (The Interview Feature) ---
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
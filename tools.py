import os
from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool

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
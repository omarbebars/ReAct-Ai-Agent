import os
from datetime import datetime
from dotenv import load_dotenv

# --- IMPORTS ---
# We stick to the imports we KNOW work in your environment
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

# Load the API Key
load_dotenv()

# --- CONFIGURATION ---

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0
)

# --- 1. DEFINE CUSTOM TOOLS DIRECTLY HERE ---

def save_to_txt(data: str):
    """Saves the provided text to a local file."""
    filename = "research_output.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)
        return f"Success: Data saved to {filename}"
    except Exception as e:
        return f"Error saving file: {e}"

# --- 2. INITIALIZE STANDARD TOOLS ---
search = DuckDuckGoSearchRun(doc_content_chars_max=100)
wikipedia = WikipediaAPIWrapper(doc_content_chars_max=100)

# --- 3. REGISTER ALL TOOLS ---
tools = [
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

# --- PROMPT ---
template = '''Answer the following questions as best you can. You have access to the following tools:

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

prompt = PromptTemplate.from_template(template)

# --- AGENT ---
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

# --- EXECUTION WITH MANUAL MEMORY ---
if __name__ == "__main__":
    print("Agent is ready! (Type 'exit' to quit)")
    
    # 1. Initialize an empty string for history
    chat_history_buffer = ""
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        try:
            # 2. Inject the history string manually into the input dictionary
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history_buffer
            })
            
            output_text = response['output']
            print(f"Agent: {output_text}")
            
            # 3. Append the interaction to our manual buffer
            chat_history_buffer += f"Human: {user_input}\nAI: {output_text}\n"
            
        except Exception as e:
            print(f"An error occurred: {e}")
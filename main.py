import os
from dotenv import load_dotenv

# --- IMPORTS ---
# We stick to the imports we KNOW work in your environment
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import get_tools, get_prompt, get_memory

# Load the API Key
load_dotenv()

# --- CONFIGURATION ---

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0
)

tools = get_tools()
prompt = get_prompt()

# --- AGENT ---
agent = create_react_agent(llm, tools, prompt)

# SQLite-backed memory (tools.py's get_memory): auto-fills {chat_history} on
# each invoke() and auto-persists to chat_history.db, so the conversation
# survives quitting and restarting this script.
memory = get_memory(session_id="cli")

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# --- EXECUTION ---
if __name__ == "__main__":
    print("Agent is ready! (Type 'exit' to quit)")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"An error occurred: {e}")
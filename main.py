import os
from dotenv import load_dotenv

# --- IMPORTS ---
# We stick to the imports we KNOW work in your environment
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import get_tools, get_prompt

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
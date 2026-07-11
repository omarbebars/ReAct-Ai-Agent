import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import agent components
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import get_tools, get_prompt

# --- Page Config ---
st.set_page_config(
    page_title="ReAct Research Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 10px;
        padding: 12px;
    }
    .main-header {
        text-align: center;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent_executor" not in st.session_state:
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
        api_key=os.getenv("Gemini_API_KEY")
    )

    tools = get_tools()
    prompt = get_prompt()

    # Create Agent
    agent = create_react_agent(llm, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 🤖 ReAct Research Agent")
    st.markdown("*Autonomous AI research assistant with web search, Wikipedia, and file persistence*")

with col2:
    st.info(f"📁 Research saves to `research_output.txt`", icon="ℹ️")

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    temperature = st.slider(
        "Temperature (creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Lower = focused, Higher = creative"
    )
    
    st.markdown("---")
    st.markdown("## 📚 Available Tools")
    st.markdown("""
    - **Search**: DuckDuckGo web search
    - **Wikipedia**: Factual lookups
    - **SaveToFile**: Persist findings
    """)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# --- Chat Display ---
st.markdown("## 💬 Conversation")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt_input := st.chat_input("Ask your research question...", key="user_input"):
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt_input,
        "avatar": "👤"
    })
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_input)
    
    # Generate response with spinner
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Researching..."):
            try:
                # Build chat history string for context
                chat_history_str = ""
                for msg in st.session_state.chat_history[:-1]:  # Exclude current input
                    chat_history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
                
                # Invoke agent
                response = st.session_state.agent_executor.invoke({
                    "input": prompt_input,
                    "chat_history": chat_history_str
                })
                
                # Handle response - could be dict or Response object
                if isinstance(response, dict):
                    output_text = response.get("output", "No response generated")
                else:
                    output_text = str(response)
                
                st.markdown(output_text)
                
                # Add to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": output_text,
                    "avatar": "🤖"
                })
                
                # Success indicator
                st.success("✅ Research complete and saved!", icon="✅")
                
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "avatar": "🤖"
                })

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    Built with <a href='https://streamlit.io'>Streamlit</a> | 
    Powered by <a href='https://ai.google.dev'>Google Gemini</a> | 
    Check <code>research_output.txt</code> for saved findings
</div>
""", unsafe_allow_html=True)

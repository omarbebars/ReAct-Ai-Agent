"""Smoke check for tools.py: run with `python test_tools.py`."""
import os
import tempfile

from tools import get_tools, get_prompt, summarize_research

tools = get_tools()
names = [t.name for t in tools]
assert names == ["Search", "Wikipedia", "SaveToFile", "SummarizeResearch"], f"unexpected tool names: {names}"

prompt = get_prompt()
expected_vars = {"tools", "tool_names", "chat_history", "input", "agent_scratchpad"}
assert set(prompt.input_variables) == expected_vars, f"unexpected prompt variables: {prompt.input_variables}"

# summarize_research must degrade gracefully (no exception, no LLM call) when
# there is no research log yet. Run from an empty temp dir so the real
# research_output.txt in the repo root is out of reach.
original_cwd = os.getcwd()
with tempfile.TemporaryDirectory() as empty_dir:
    os.chdir(empty_dir)
    try:
        result = summarize_research("anything")
    finally:
        os.chdir(original_cwd)
assert "No research history found" in result, f"unexpected no-file message: {result}"

print("OK: tools.py exposes the expected tools, prompt template, and summarizer fallback.")

"""Smoke check for optimize_agent.py's placeholder guard. Run: python test_optimize_agent.py"""
from optimize_agent import validate_prompt_template, REQUIRED_PLACEHOLDERS

ok, missing = validate_prompt_template("{tools} {tool_names} {chat_history} {input} {agent_scratchpad}")
assert ok and missing == [], f"expected valid template to pass, got missing={missing}"

ok, missing = validate_prompt_template("{tools} {tool_names} {input}")
assert not ok and set(missing) == {"{chat_history}", "{agent_scratchpad}"}, f"expected missing placeholders detected, got {missing}"

assert len(REQUIRED_PLACEHOLDERS) == 5

print("OK: validate_prompt_template correctly detects missing placeholders.")

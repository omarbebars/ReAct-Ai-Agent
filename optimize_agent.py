"""
GEPA prompt optimization for the ReAct agent, using deepeval's built-in
PromptOptimizer + GEPA algorithm (deepeval.optimizer) rather than a hand-rolled
reflect-and-mutate loop.

Runs the existing LangChain agent (tools.py's get_tools()/REACT_PROMPT_TEMPLATE)
against a 30-golden dataset spanning 10 known ReAct failure categories. Grading
uses Gemini (compliance metric); reflection/mutation use OpenRouter's free tier
so no OpenAI key is needed.

This is an OFFLINE script: it never touches tools.py. It writes the best
candidate prompt to optimized_prompt.txt for human review.

Requires: pip install deepeval
Env vars: Gemini_API_KEY, OPENROUTER_API_KEY (see .env.example)

Run: python optimize_agent.py
"""
import os
from dotenv import load_dotenv
from deepeval.dataset import Golden
from deepeval.metrics import GEval, PatternMatchMetric
from deepeval.test_case import SingleTurnParams
from deepeval.models import GeminiModel, OpenRouterModel
from deepeval.prompt import Prompt
from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.algorithms import GEPA
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from tools import get_tools, REACT_PROMPT_TEMPLATE

load_dotenv()

MAX_AGENT_ITERATIONS = 6
REQUIRED_PLACEHOLDERS = ["{tools}", "{tool_names}", "{chat_history}", "{input}", "{agent_scratchpad}"]

GEMINI_API_KEY = os.getenv("Gemini_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Free-tier OpenRouter reasoning model for GEPA's reflection/mutation steps (no OpenAI key
# needed). gpt-oss-20b:free supports OpenRouter's "reasoning" param, which gives GEPA's
# reflection step an explicit chain-of-thought over *why* a candidate prompt failed before
# it proposes a mutation -- should yield sharper rewrites than a non-reasoning model.
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "openai/gpt-oss-20b:free")
# "reasoning" isn't a real OpenAI chat-completions param, so it must go through
# extra_body (the openai SDK's escape hatch for provider-specific fields) rather
# than as a top-level generation_kwargs entry, or create() raises a TypeError.
OPENROUTER_REASONING_KWARGS = {"extra_body": {"reasoning": {"enabled": True}}}

AGENT_LLM = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, api_key=GEMINI_API_KEY)
JUDGE_MODEL = GeminiModel(model="gemini-2.5-flash", api_key=GEMINI_API_KEY, temperature=0)
REFLECTION_MODEL = OpenRouterModel(
    model=OPENROUTER_MODEL_NAME, api_key=OPENROUTER_API_KEY, temperature=0,
    generation_kwargs=OPENROUTER_REASONING_KWARGS,
)
MUTATION_MODEL = OpenRouterModel(
    model=OPENROUTER_MODEL_NAME, api_key=OPENROUTER_API_KEY, temperature=0.7,
    generation_kwargs=OPENROUTER_REASONING_KWARGS,
)


# --- Golden dataset: 30 queries across 10 ReAct failure categories ---
GOLDENS = [
    # --- CATEGORY 1: Chronological Hallucination / Recency Bias ---
    Golden(
        input="What are the latest embedded systems research papers published by HTW Berlin in 2026?",
        expected_output="The agent must use the web search tool to find 2026 data and not hallucinate fake papers or state that 2026 hasn't happened yet.",
    ),
    Golden(
        input="Who won the UEFA Champions League in 2026?",
        expected_output="The agent must recognize that this requires real-time search and use the DuckDuckGo tool rather than stating the year is in the future.",
    ),
    Golden(
        input="What are the current tech layoffs or hiring trends in Berlin's automotive software sector right now?",
        expected_output="The agent must use the web search tool to find live news rather than relying on its static pre-training data.",
    ),

    # --- CATEGORY 2: Missing State / File Persistence ---
    Golden(
        input="Research the exact camera specifications of the iPhone 15 Pro Max and save the results to a file.",
        expected_output="The agent must first use Web Search or Wikipedia, and then explicitly call the SaveToFile tool before providing the Final Answer.",
    ),
    Golden(
        input="Find the hardware specs of the Lenovo ThinkPad T14 and make sure they are written to disk.",
        expected_output="The agent must look up the specs and explicitly execute SaveToFile to write the configuration to disk before ending.",
    ),
    Golden(
        input="Look up the latest version of the deepeval Python library and save that version number to a local file.",
        expected_output="The agent must search for the current library info and invoke the SaveToFile tool to store the result.",
    ),

    # --- CATEGORY 3: Multi-step Reasoning / Comparison Split ---
    Golden(
        input="Compare the current weather today in Berlin with the weather in Stockholm.",
        expected_output="The agent must execute two separate web searches (one for Berlin, one for Stockholm) before generating the Final Answer.",
    ),
    Golden(
        input="Who is older: Elon Musk or Sam Altman? Verify both birthdays.",
        expected_output="The agent must execute separate tool actions to look up both distinct individuals before making the mathematical comparison.",
    ),
    Golden(
        input="Compare the academic entry requirements of HTW Berlin versus TU Berlin for Computer Engineering.",
        expected_output="The agent must perform a search or lookup for HTW Berlin and a separate search for TU Berlin before synthesizing the final output.",
    ),

    # --- CATEGORY 4: Syntax Breakage / Code-Block Bleeding ---
    Golden(
        input="Generate a Mermaid.js diagram code showing a standard ReAct loop and save it.",
        expected_output="The agent must output valid Mermaid.js code and successfully pass it into the SaveToFile tool without breaking the ReAct text formatting.",
    ),
    Golden(
        input="Write a Python script that implements a basic template matching function and save it to 'template.py'.",
        expected_output="The agent must handle the raw Python string contents cleanly inside the Action Input payload without breaking the main ReAct parser tags.",
    ),
    Golden(
        input="Create an n8n JSON workflow snippet for an email webhook and dump it to a file.",
        expected_output="The agent must pass raw nested JSON data into the SaveToFile action without hitting JSON string escaping syntax breaks.",
    ),

    # --- CATEGORY 5: Tool Misallocation / Lazy Selection ---
    Golden(
        input="What is the exact founding date and history of Mercedes-Benz Tech Innovation according to Wikipedia?",
        expected_output="The agent must strictly select the Wikipedia tool, not the standard web search tool.",
    ),
    Golden(
        input="Look up the historical origin of the word 'robotics' using your encyclopedia tool.",
        expected_output="The agent must choose the Wikipedia tool for this deep historical definition rather than running a noisy general web search.",
    ),
    Golden(
        input="According to Wikipedia, what was the original name of the Hochschule für Technik und Wirtschaft Berlin?",
        expected_output="The agent must target Wikipedia directly to extract this specific historical fact.",
    ),

    # --- CATEGORY 6: Graceful Error Handling & Fallbacks ---
    Golden(
        input="Search Wikipedia for 'Jochum Medizintechnik GmbH' and summarize what you find.",
        expected_output="If the Wikipedia tool returns no results, the agent must observe the failure and gracefully pivot to the Web Search tool instead of crashing or giving up.",
    ),
    Golden(
        input="Look up 'CustomTurtleBot999' on Wikipedia and tell me what it is.",
        expected_output="When Wikipedia returns a 'Page not found' or empty response, the agent must seamlessly fall back to Web Search to find the niche topic.",
    ),
    Golden(
        input="Search Wikipedia for a highly specific local company called 'Kamps Berlin' and summarize it.",
        expected_output="The agent must catch an empty Wikipedia search response and immediately execute a Web Search step to find the business profile.",
    ),

    # --- CATEGORY 7: Premature Termination / Skimming ---
    Golden(
        input="What are the step-by-step instructions to set up an n8n workflow for processing GitHub pull requests?",
        expected_output="The agent should perform multiple searches if necessary to retrieve actual, actionable steps, rather than returning a generic one-sentence summary from the first search result.",
    ),
    Golden(
        input="Give me a comprehensive, detailed guide on how to configure a local LLM on an Ubuntu machine.",
        expected_output="The agent must pull multiple rich operational steps across its loop iterations rather than providing a superficial three-bullet-point summary.",
    ),
    Golden(
        input="Provide a complete, detailed walkthrough for debugging a DSPy prompt optimization pipeline.",
        expected_output="The agent must dig deeper into the technical details across multiple thoughts before compiling its detailed Final Answer.",
    ),

    # --- CATEGORY 8: Complex Parameter / Argument Passing ---
    Golden(
        input="Save a file named 'lenovo_t14_specs.txt' containing the hardware specs of the Lenovo ThinkPad T14.",
        expected_output="The agent must correctly format the Action Input to pass both the specific filename and the content to the save tool without escaping errors.",
    ),
    Golden(
        input="Write a file named 'hello.sh' that echoes 'Hello World' and save it to disk.",
        expected_output="The agent must accurately extract and isolate both the exact string filename parameter and file body contents without text overlap.",
    ),
    Golden(
        input="Log the current stock price of Apple into a file called 'apple_tracker.txt'.",
        expected_output="The agent must fetch the live information first and correctly split parameters to execute the file-save action cleanly.",
    ),

    # --- CATEGORY 9: Conversational Distraction / Token Noise ---
    Golden(
        input="Hi, I'm working on a Python script. Can you search for the latest documentation on the 'deepeval' library?",
        expected_output="The agent must ignore the conversational filler and immediately execute a web search for 'deepeval Python library documentation'.",
    ),
    Golden(
        input="Good morning! I have an exam coming up. Can you search for past exam questions on verification and validation algorithms?",
        expected_output="The agent must avoid replying with casual banter and focus entirely on executing its search actions to fulfill the query target.",
    ),
    Golden(
        input="Wow, this terminal looks great. Hey, can you find out who currently maintains the LangChain framework?",
        expected_output="The agent must bypass the initial exclamation and instantly route its focus into a valid search or lookup action.",
    ),

    # --- CATEGORY 10: Loop / Token Lock Prevention ---
    Golden(
        input="Find information on something that doesn't exist like 'Zxyvba123' and give me a full breakdown.",
        expected_output="The agent must stop after failing to find information across 2-3 attempts instead of spinning infinitely in an observation loop.",
    ),
    Golden(
        input="Search for the latest 2026 firmware updates for an unreleased model called Lenovo ThinkPad T18.",
        expected_output="The agent must gracefully declare it cannot find reliable information after a couple of targeted tool executions instead of repeating the same search indefinitely.",
    ),
    Golden(
        input="Track down the code repository of an internal private module named 'local_utils_auth_v5'.",
        expected_output="The agent must gracefully fail its search actions and close the loop with a factual final report rather than looping endlessly.",
    ),
]

# --- Metrics: decoupled per behavior so GEPA's reflection model gets a clear,
# independent signal per failure mode instead of one blended score. Each
# metric is graded against the full trace (see model_callback), not just the
# Final Answer text, since tool-selection/looping/syntax all live in the trace.

# Metric 1: ReAct text syntax enforcer -- the trace must show at least one
# well-formed Action/Action Input pair, not just prose.
react_syntax_metric = PatternMatchMetric(
    pattern=r"[\s\S]*Action:\s*[^\n]+\s*Action Input:\s*[\s\S]*",
    ignore_case=True,
)

# Metric 2: Tool routing accuracy (Wikipedia vs Search vs SaveToFile).
tool_routing_metric = GEval(
    name="ToolRoutingAndGuardrail",
    criteria="Evaluate whether the agent made the correct tool choices shown in the trace. "
             "Penalize using general web search when Wikipedia was explicitly requested, "
             "or failing to call SaveToFile when the input demanded persistence.",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.7,
)

# Metric 3: Reasoning depth / loop prevention.
reasoning_depth_metric = GEval(
    name="ReasoningDepthAndLoopPrevention",
    criteria="Evaluate if the agent thoroughly unpacked comparative or multi-part queries by making "
             "multiple distinct tool calls shown in the trace. Penalize premature answers that only "
             "cover a single perspective, as well as repetitive tool calls that indicate a stuck loop.",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.7,
)

# Metric 4: Temporal grounding -- current baseline year is 2026.
factual_grounding_metric = GEval(
    name="TemporalGuardrailsAndGrounding",
    criteria="Evaluate if the agent anchors its answer in live search/Wikipedia data for current or "
             "recent events, using a current temporal baseline of 2026. Severely penalize hallucinated "
             "entries or refusal to search on the grounds that 2026 is a future date.",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.7,
)

METRICS = [react_syntax_metric, tool_routing_metric, reasoning_depth_metric, factual_grounding_metric]


def validate_prompt_template(template: str):
    """Pure check: a rewritten template must keep every placeholder the agent
    depends on, or it will crash create_react_agent()."""
    missing = [p for p in REQUIRED_PLACEHOLDERS if p not in template]
    return (len(missing) == 0, missing)


def build_agent_executor(prompt_template: str) -> AgentExecutor:
    prompt = PromptTemplate.from_template(prompt_template)
    agent = create_react_agent(AGENT_LLM, get_tools(), prompt)
    return AgentExecutor(
        agent=agent,
        tools=get_tools(),
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=MAX_AGENT_ITERATIONS,
        return_intermediate_steps=True,
    )


def model_callback(prompt: Prompt, golden: Golden) -> str:
    """Runs the actual LangChain ReAct agent for a GEPA-proposed prompt candidate.

    GEPA only manages the prompt text; the ReAct loop itself (tool dispatch,
    {tools}/{tool_names}/{agent_scratchpad} filling) still runs through the
    existing AgentExecutor, so this ignores Prompt.interpolate() and instead
    rebuilds the agent from the candidate's text_template directly.

    Returns the full tool-call trace, not just the Final Answer text --
    react_syntax_metric, tool_routing_metric, and reasoning_depth_metric all
    need to see which tools were called and how many cycles ran, which only
    exists in intermediate_steps, not in response['output'] alone.
    """
    ok, missing = validate_prompt_template(prompt.text_template)
    if not ok:
        return f"[invalid candidate prompt, missing placeholders {missing}]"

    try:
        agent_executor = build_agent_executor(prompt.text_template)
        response = agent_executor.invoke({"input": golden.input, "chat_history": ""})
        output = response.get("output", "")
        steps = response.get("intermediate_steps", [])
        trace = "; ".join(f"{action.tool}({action.tool_input!r}) -> {str(obs)[:200]}" for action, obs in steps)
    except Exception as e:
        return f"[agent raised an exception before a Final Answer: {e}]"

    return f"Tool calls: {trace or '(none)'}\nFinal Answer: {output}"


def main():
    gepa = GEPA(
        reflection_model=REFLECTION_MODEL,
        mutation_model=MUTATION_MODEL,
        iterations=7,
        pareto_size=6,
        minibatch_size=8,
        patience=2,
    )
    # PromptOptimizer's own optimizer_model (distinct from GEPA's reflection_model/
    # mutation_model above) builds the internal Scorer/Rewriter eagerly at
    # construction time, defaulting to OpenAI's GPTModel if left unset -- which
    # crashes without OPENAI_API_KEY before GEPA.execute() ever runs and gets a
    # chance to override them with reflection_model/mutation_model. Pass our
    # OpenRouter model explicitly so no OpenAI key is required anywhere.
    optimizer = PromptOptimizer(
        algorithm=gepa, metrics=METRICS, model_callback=model_callback, optimizer_model=REFLECTION_MODEL,
    )

    starting_prompt = Prompt(text_template=REACT_PROMPT_TEMPLATE)
    optimized_prompt = optimizer.optimize(prompt=starting_prompt, goldens=GOLDENS)

    print("Optimization report:", optimizer.optimization_report)

    ok, missing = validate_prompt_template(optimized_prompt.text_template)
    if not ok:
        print(f"GEPA's result is missing required placeholders {missing}; refusing to write it out.")
        return

    if optimized_prompt.text_template.strip() == REACT_PROMPT_TEMPLATE.strip():
        print("No change from baseline REACT_PROMPT_TEMPLATE.")
        return

    with open("optimized_prompt.txt", "w", encoding="utf-8") as f:
        f.write(optimized_prompt.text_template)
    print("Written to optimized_prompt.txt -- review before copying into tools.py's REACT_PROMPT_TEMPLATE.")


if __name__ == "__main__":
    main()

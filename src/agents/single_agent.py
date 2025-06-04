# src/agents/single_agent.py
"""
Single-agent LangGraph pipeline for the clinical-reasoning benchmark.

The graph contains **one** thinking LLM plus a single tool node
(`Retrieve Results`) that surfaces physical-exam findings, labs,
and imaging when the model requests them.  
Execution proceeds like:

    ┌───────────────┐ “agent”  (call_model → LLM)
    │   Thought     │
    │   Action      │  ──▶ retrieve? ──┐
    │   Action Input│                  │
    └───────────────┘                  │
            ▲                          ▼
            │                 ┌────────────────┐
            └─────────────────│   tools node   │
                              │ RetrieveResults│
                              └────────────────┘

The run terminates when either
* the agent produces **Format-2** output (Final Diagnosis / Treatment), or
* the hard `MAX_ITERATIONS` cap is hit.

Key sections
------------
- **Regex pre-compiles** Fast parsers for Format-1 / Format-2 / Lab-blocks
- **AgentState** LangGraph typed state (messages + counters)
- **call_model()** Wraps the main LLM, parses its output, attaches tool calls
- **should_continue()** Conditional edge deciding whether to loop or stop
- **build_graph()** Returns a compiled `StateGraph` ready for execution
"""


import json
import re
from typing import Any, Dict, List

from langgraph.graph import (
    MessagesState, 
    StateGraph, 
    END
)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from ..prompts   import SYSTEM_PROMPT, DIAGNOSIS_PROMPT
from ..config import CONFIG
from .retrieve_results import RetrieveResults, make_retrieve_node


# ────────────────────────────────────────────────────────────────────────────
# Agent State
# ────────────────────────────────────────────────────────────────────────────

class AgentState(MessagesState):
    """Holds the conversation messages, plus patient_id & iteration count."""
    summary: str
    patient_id: str
    iteration: int = 0


# ────────────────────────────────────────────────────────────────────────────
# LLM Parsing
# ────────────────────────────────────────────────────────────────────────────

def parse_llm_response(response: str, has_lab_results: bool = False) -> Dict[str, str]:
    """
    Parses the LLM response to extract either Format 1 or Format 2 responses.
    Only looks for Lab Interpretation if has_lab_results is True.
    """
    if not re.search(r"^Thought:", response, re.MULTILINE | re.IGNORECASE):
        # If response doesn't start with "Thought:" but has content,
        # restructure it to fit the expected format
        if re.search(r"Action:", response, re.MULTILINE | re.IGNORECASE):
            response = "Thought: Analyzing the case.\n" + response
        else:
            # If there's no action either, treat entire text as thought
            response = f"Thought: {response.strip()}\nAction: physical examination\nAction Input: Full physical exam"

    if not isinstance(response, str):
        response = str(response or "")
    response = re.sub(r"</?think>", "", response, flags=re.I).strip()

    # Try Format 1 first (information gathering)
    format1_pattern = re.compile(
        r"(?:^|\s*)Thought:\s*(?P<thought>.*?)\s*\n"
        r"Action:\s*(?P<action>[^\n]*\S[^\n]*)\s*\n"   # ← at least one non‑space
        r"Action Input:\s*(?P<action_input>.*?)(?=\s*$|\s*\n\s*\n)",
        re.DOTALL | re.MULTILINE
    )
    
    # Try Format 2 (final diagnosis)
    format2_pattern = re.compile(
    r"(?:(?:^|\s*)Thought:\s*(?P<thought>.*?)\s*\n)?"
    r"Final Diagnosis\s*(?:\*\*?)?\s*(?:\((?:ranked|Ranked)\))?\s*:?[\s\*]*\n"
    r"(?P<diagnosis>.*?)\n"
    r"Treatment\s*:?\s*(?P<treatment>.*?)(?=\s*$|\n\s*\n)",
    re.IGNORECASE | re.DOTALL
    )
    
    # If we're expecting lab results, try to parse with Lab Interpretation
    if has_lab_results:
        lab_pattern = re.compile(
            r"(?:^|\s*)Thought:\s*(?P<thought>.*?)\s*\n"
            r"Lab Interpretation:\s*(?P<lab_interpretation>\{.*?\})\s*\n"
            r"Action:\s*(?P<action>.*?)\s*\n"
            r"Action Input:\s*(?P<action_input>.*?)(?=\s*$|\s*\n\s*\n)",
            re.DOTALL | re.MULTILINE
        )
        match = lab_pattern.search(response)
        if match:
            lab_interpretation = match.group("lab_interpretation")
            try:
                # Try to parse and reformat the JSON to ensure it's valid
                lab_json = json.loads(lab_interpretation)
                lab_interpretation = json.dumps(lab_json, indent=2)
            except json.JSONDecodeError:
                lab_interpretation = ""
            
            return {
                "thought": (match.group("thought") or "").strip(),
                "lab_interpretation": lab_interpretation,
                "action": match.group("action").strip().lower(),
                "action_input": match.group("action_input").strip(),
                "tool_call": True
            }
    
    # Check Format 1 (without Lab Interpretation)
    match = format1_pattern.search(response)
    if match:
        return {
            "thought": (match.group("thought") or "").strip(),
            "lab_interpretation": "",
            "action": match.group("action").strip().lower(),
            "action_input": match.group("action_input").strip(),
            "tool_call": True
        }
    
    # Check Format 2
    match = format2_pattern.search(response)
    if match:
        return {
            "thought": (match.group("thought") or "").strip(),
            "lab_interpretation": "",  # Empty for final diagnosis format
            "action": "",
            "action_input": "",
            "diagnosis": match.group("diagnosis").strip(),
            "treatment": match.group("treatment").strip(),
            "tool_call": False
        }
    
    # If neither format matches, return empty format 1 structure
    return {
        "thought": "",
        "lab_interpretation": "",
        "action": "",
        "action_input": response.strip(),
        "tool_call": True
    }


def extract_tool_calls(response_dict: Dict[str, Any], patient_id: str) -> List[Dict[str, Any]]:
    if response_dict.get("tool_call") and response_dict.get("action"):
        return [{
            "name": "Retrieve Results",
            "args": {"patient_id": patient_id, "response_dict": response_dict},
            "id":   "0",
        }]
    return []


def call_model(state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    messages   = state["messages"]
    patient_id = state["patient_id"]
    iteration  = state["iteration"]

    # detect if last AI asked for labs, and we now have a ToolMessage
    has_lab = (
        len(messages) >= 2
        and isinstance(messages[-1], ToolMessage)
        and isinstance(messages[-2], AIMessage)
        and parse_llm_response(messages[-2].content, False).get("action") == "laboratory tests"
    )

    # pick system vs diagnosis prompt
    sys_msg = SystemMessage(content=DIAGNOSIS_PROMPT) if iteration >= CONFIG.runtime.max_iterations \
              else SystemMessage(content=SYSTEM_PROMPT)
    prompt  = [sys_msg] + messages

    raw = llm.invoke(prompt)
    pr  = parse_llm_response(raw, has_lab)

    # format back into a single string
    if "diagnosis" in pr:
        formatted = raw
    else:
        parts = [f"Thought: {pr['thought']}"]
        if has_lab:
            parts.append(f"Lab Interpretation: {pr['lab_interpretation']}")
        parts.append(f"Action: {pr['action']}")
        parts.append(f"Action Input: {pr['action_input']}")
        formatted = "\n".join(parts)

    ai_msg = AIMessage(content=formatted)
    ai_msg.tool_calls = extract_tool_calls(pr, patient_id)
    return {"messages": [ai_msg], "iteration": iteration + 1}


# ────────────────────────────────────────────────────────────────────────────
# Graph helpers
# ────────────────────────────────────────────────────────────────────────────

def should_continue(state: AgentState):
    messages = state["messages"]
    iteration = state["iteration"]
    last_message = messages[-1]
    # Stop if we reach max iteration
    if iteration > CONFIG.runtime.max_iterations or not last_message.tool_calls:
        return "end"
    else:
        return "continue"



def build_graph(main_llm: Any, matcher_llm: Any, patient_data: dict) -> StateGraph:
    retrieve = RetrieveResults(
        name="Retrieve Results",
        patient_data=patient_data,
        matcher_llm=matcher_llm
    )
    tools = {"Retrieve Results": retrieve}

    g = StateGraph(AgentState)
    g.add_node("agent", lambda s: call_model(s, main_llm))
    g.add_node("tools", make_retrieve_node(tools))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    g.add_edge("tools", "agent")
    return g.compile()

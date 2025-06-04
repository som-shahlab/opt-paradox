# src/agents/multi_agent.py
"""
**Three-agent LangGraph pipeline**

This variant breaks the reasoning workflow into specialised stages:

1. **Info Gathering Agent** – asks for physical-exam / labs / imaging  
2. **Interpretation Agent** – looks at raw tool output and explains it  
3. **Diagnosis Agent** – delivers the final diagnosis & treatment

The single shared tool is again **Retrieve Results**.  
Control flow:

    InfoGathering ──┐─▶ Tools ──┐─▶ Interpretation
          ▲         │          │
          │         └──────────┘
          └────────────────────────▶ Diagnosis (terminal)

Edges
-----
* `sufficient_info` – from InfoGathering decide “continue” (loop) vs “diagnosis”.
  It checks iteration cap **or** an explicit `action: done`.
* `should_interpret` – from Tools decide “interpret” vs loop back.  
  Fires only when the tool just returned *laboratory tests*.
"""


import re
import json
from typing import Any, Dict, List

from langgraph.graph import (
    MessagesState, 
    StateGraph, 
    END
)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from ..prompts   import (
    INFO_GATHERING_PROMPT,
    INTERPRETATION_PROMPT,
    DIAGNOSIS_PROMPT
)
from ..config import CONFIG

from .retrieve_results  import RetrieveResults, make_retrieve_node


# -----------------------------------------------------------------------------
#  Custom Message Types
# -----------------------------------------------------------------------------
class InfoGatheringMessage(AIMessage):
    pass

class InterpretationMessage(AIMessage):
    pass

class DiagnosisMessage(AIMessage):
    pass


class AgentState(MessagesState):
    """Holds the conversation messages, plus patient_id & iteration count."""
    summary: str
    patient_id: str
    iteration: int = 0


# -----------------------------------------------------------------------------
#  Agent Callables
# -----------------------------------------------------------------------------
def parse_info_gathering_response(response: str) -> Dict[str, Any]:
    """
    Parses the Info Gathering LLM output into thought/action/action_input.
    """
    if not re.search(r"^Thought:", response, re.MULTILINE | re.IGNORECASE):
        if re.search(r"Action:", response, re.MULTILINE | re.IGNORECASE):
            response = "Thought: Analyzing the case.\n" + response
        else:
            response = (
                f"Thought: {response.strip()}\n"
                "Action: physical examination\n"
                "Action Input: Full physical exam"
            )

    response = re.sub(r"</?think>", "", str(response), flags=re.I).strip()

    pattern = re.compile(
        r"(?:^|\s*)Thought:\s*(?P<thought>.*?)\s*\n"
        r"Action:\s*(?P<action>[^\n]*\S[^\n]*)\s*\n"
        r"Action Input:\s*(?P<action_input>.*?)(?=\s*$|\s*\n\s*\n)",
        re.DOTALL | re.MULTILINE
    )
    m = pattern.search(response)
    if m:
        return {
            "thought":      m.group("thought").strip(),
            "action":       m.group("action").strip().lower(),
            "action_input": m.group("action_input").strip(),
            "tool_call":    True
        }

    # fallback
    return {
        "thought":      "",
        "action":       "",
        "action_input": response.strip(),
        "tool_call":    True
    }


def extract_tool_calls(response_dict: Dict[str, Any], patient_id: str) -> List[Dict[str, Any]]:
    if response_dict.get("tool_call") and response_dict.get("action"):
        return [{
            "name": "Retrieve Results",
            "args": {"patient_id": patient_id, "response_dict": response_dict},
            "id":   "0",
        }]
    return []


def gather_info(state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    Info-Gathering Agent node: calls LLM, parses response, attaches tool_calls.
    """
    messages   = state["messages"]
    patient_id = state["patient_id"]

    sys_msg         = SystemMessage(content=INFO_GATHERING_PROMPT)
    raw_response    = llm.invoke([sys_msg] + messages)
    pr              = parse_info_gathering_response(raw_response)
    formatted       = (
        f"Thought: {pr['thought']}\n"
        f"Action: {pr['action']}\n"
        f"Action Input: {pr['action_input']}"
    )
    tool_calls      = extract_tool_calls(pr, patient_id)
    agent_response  = InfoGatheringMessage(content=formatted)
    agent_response.tool_calls = tool_calls


    return {
        "messages":            [agent_response],
        "iteration":           state["iteration"] + 1,
    }


def interpret_results(state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    Interpretation Agent node: takes tool output messages and interprets them.
    """
    messages = state["messages"]
    sys_msg  = SystemMessage(content=INTERPRETATION_PROMPT)

    raw_response   = llm.invoke([sys_msg] + messages)
    agent_response = InterpretationMessage(content=raw_response)

    return {"messages": [agent_response]}


def give_diagnosis(state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    Diagnosis Agent node: makes the final diagnosis.
    """
    # Don't include the last message from InfoGatheringAgent 
    # To try and reduce bias from Info gathering Agent
    messages = state["messages"][:-1]
    sys_msg  = SystemMessage(content=DIAGNOSIS_PROMPT)

    raw_response   = llm.invoke([sys_msg] + messages)
    agent_response = DiagnosisMessage(content=raw_response)


    return {"messages": [agent_response]}


# -----------------------------------------------------------------------------
#  Conditional Edges
# -----------------------------------------------------------------------------
def sufficient_info(state: AgentState) -> str:
    """
    After InfoGathering, decide whether to go to Tools or straight to Diagnosis.
    """
    last = state["messages"][-1]
    if state["iteration"] >= CONFIG.runtime.max_iterations  or not getattr(last, "tool_calls", []):
        return "diagnosis"
    args = last.tool_calls[0]["args"]["response_dict"]
    if args.get("action") == "done":
        return "diagnosis"
    return "continue"


def should_interpret(state: AgentState) -> str:
    """
    After Tools, decide whether to go to Interpretation or back to InfoGathering.
    """
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        # RetrieveResults tool encodes action in the result dict
        action = last.additional_kwargs.get("action")
        if action == "laboratory tests":
            return "interpret"
    return "continue"


# -----------------------------------------------------------------------------
#  Graph Builder
# -----------------------------------------------------------------------------
def build_graph(
    info_llm: Any,
    interpretation_llm: Any,
    matcher_llm: Any,
    diagnosis_llm: Any,
    patient_data: Dict[str, Any],
) -> StateGraph:
    retrieve_tool = RetrieveResults(
        name="Retrieve Results",
        patient_data=patient_data,
        matcher_llm=matcher_llm
    )
    tools = {"Retrieve Results": retrieve_tool}

    g = StateGraph(AgentState)
    g.set_entry_point("Info Gathering Agent")

    g.add_node("Info Gathering Agent", lambda s: gather_info(s, info_llm))
    g.add_node("Tools",                 make_retrieve_node(tools))
    g.add_node("Interpretation Agent",  lambda s: interpret_results(s, interpretation_llm))
    g.add_node("Diagnosis Agent",       lambda s: give_diagnosis(s, diagnosis_llm))

    g.add_conditional_edges(
        "Info Gathering Agent",
        sufficient_info,
        {"continue": "Tools", "diagnosis": "Diagnosis Agent"}
    )
    g.add_conditional_edges(
        "Tools",
        should_interpret,
        {"interpret": "Interpretation Agent", "continue": "Info Gathering Agent"}
    )
    g.add_edge("Interpretation Agent", "Info Gathering Agent")
    g.add_edge("Diagnosis Agent", END)

    return g.compile()

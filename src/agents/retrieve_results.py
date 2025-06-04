# src/agents/retrieve_results.py
"""
`RetrieveResults` tool and helper node for LangGraph execution
--------------------------------------------------------------

* **RetrieveResults** – a `BaseTool` that returns patient-specific data:
  - *physical examination* text
  - *laboratory tests* (passed through a matcher-LLM so we only return the
    tests the agent asked for)
  - *imaging studies* (ditto, summarised by the matcher-LLM)

The tool’s `action` field lets downstream nodes know **what kind** of
result they’re looking at.

* **make_retrieve_node(tools)** – small factory that wraps one (or more)
  tool instances into a LangGraph node.  
  It scans the last AIMessage for `tool_calls`, invokes the corresponding
  tool(s), and emits a list of `ToolMessage`s ready to be appended to the
  conversation state.

If you introduce extra tools, create a new node factory or extend this one;
its JSON schema is tightly coupled to the output of `RetrieveResults`.
"""


import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import BaseTool

from ..prompts import LABS_MATCHER_PROMPT, IMAGING_MATCHER_PROMPT


class RetrieveResults(BaseTool):
    """
    Tool for retrieving patient data:
      - physical examination
      - laboratory tests
      - imaging studies
    """
    name = "Retrieve Results"
    description = "Retrieves physical exam, lab, or imaging results."
    patient_data: Dict[str, Any]
    matcher_llm: Any

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pid = inputs["patient_id"]
        resp = inputs["response_dict"]
        action = resp.get("action", "").lower()
        if action == "physical examination":
            return self._physical(pid)
        if action == "laboratory tests":
            return self._labs(pid, resp.get("action_input", ""))
        if action == "imaging":
            return self._imaging(pid, resp.get("action_input", ""))
        # invalid action: prompt model to use correct format
        return {"action": None, "result": "Please use the correct format to request Physical Examination, Laboratory Tests, or Imaging."}

    def _physical(self, pid: str) -> Dict[str, Any]:
        findings = self.patient_data.get(pid, {})\
            .get("Physical Examination", "No physical exam data.")
        return {"action": "physical examination", "result": findings}

    def _labs(self, pid: str, requested: str) -> Dict[str, Any]:
        labs = self.patient_data.get(pid, {})\
            .get("Laboratory Tests", {})
        available = ", ".join(f"{k}: {v}" for k, v in labs.items())
        prompt = LABS_MATCHER_PROMPT.format(
            requested_tests=requested,
            available_tests=available
        )
        text = self.matcher_llm.invoke([HumanMessage(content=prompt)])\
            .replace("\n", " ")
        return {"action": "laboratory tests", "result": text}

    def _imaging(self, pid: str, requested: str) -> Dict[str, Any]:
        studies = self.patient_data.get(pid, {})\
            .get("Radiology", [])
        if not studies:
            return {"action": "imaging", "result": "No imaging data."}
        summary = ", ".join(
            f"{s.get('Exam Name','?')} ({s.get('Modality','?')}): "
            f"{s.get('Report','').replace(chr(10), ' ')}"
            for s in studies
        )
        prompt = IMAGING_MATCHER_PROMPT.format(
            requested_imaging=requested,
            available_imaging=summary
        )
        text = self.matcher_llm.invoke([HumanMessage(content=prompt)])\
            .replace("\n", " ")
        return {"action": "imaging", "result": text}
    
    
def make_retrieve_node(tools: Dict[str, BaseTool]):
    """
    Return a LangGraph node that executes *Retrieve Results* tool-calls found
    on the last AIMessage.

    NOTE: this node assumes the call schema produced by `RetrieveResults`;
    if you add new tools you’ll need a separate factory or a generic router.
    """
    def node(state) -> Dict[str, List[ToolMessage]]:
        outputs: List[ToolMessage] = []
        last_msg = state["messages"][-1]
        if not hasattr(last_msg, "tool_calls"):
            return {"messages": []}
        for call in last_msg.tool_calls:
            tool = tools.get(call["name"])
            if not tool:
                continue
            result = tool.invoke({"inputs": call["args"]})
            outputs.append(ToolMessage(
                content=json.dumps(result["result"]),
                name=call["name"],
                tool_call_id=call["id"],
                additional_kwargs={"action": result["action"]}
            ))
        return {"messages": outputs}
    return node


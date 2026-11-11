"""The LangGraph workflow.

    START -> guardrail --blocked--> synthesizer -> END
                 |
                 v
             supervisor -> [specialist]* -> synthesizer -> END

The supervisor writes a `plan` (ordered list of specialists). After each
specialist runs, `route_next` sends control to the next one in the plan, so a
single message like "compare the 17 and 17 Pro, and what's the return policy?"
runs product_expert then support, and the synthesizer merges both.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.cart import cart
from app.agents.guardrail import guardrail_node
from app.agents.inventory import inventory
from app.agents.order_tracking import order_tracking
from app.agents.product_expert import product_expert
from app.agents.recommender import recommender
from app.agents.state import StoreState
from app.agents.supervisor import supervisor_node
from app.agents.support import support
from app.agents.synthesizer import synthesizer_node
from app.agents.trade_in import trade_in

SPECIALISTS = [product_expert, recommender, inventory, support, trade_in, cart, order_tracking]
SPECIALIST_NAMES = [s.agent_name for s in SPECIALISTS]


def route_next(state: StoreState) -> str:
    plan = state.get("plan") or []
    return plan[0] if plan else "synthesizer"


def after_guardrail(state: StoreState) -> str:
    return "synthesizer" if state.get("blocked") else "supervisor"


def build_graph(checkpointer=None):
    g = StateGraph(StoreState)
    g.add_node("guardrail", guardrail_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("synthesizer", synthesizer_node)
    for spec in SPECIALISTS:
        g.add_node(spec.agent_name, spec)

    targets = SPECIALIST_NAMES + ["synthesizer"]
    g.add_edge(START, "guardrail")
    g.add_conditional_edges("guardrail", after_guardrail, ["supervisor", "synthesizer"])
    g.add_conditional_edges("supervisor", route_next, targets)
    for name in SPECIALIST_NAMES:
        g.add_conditional_edges(name, route_next, targets)
    g.add_edge("synthesizer", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


graph = build_graph()


async def chat(message: str, session_id: str) -> dict:
    """Run one conversational turn. State for `session_id` persists between calls."""
    result = await graph.ainvoke(
        {"user_input": message, "session_id": session_id, "findings": None, "trace": None},
        config={"configurable": {"thread_id": session_id}},
    )
    return {
        "answer": result["answer"],
        "intents": result.get("intents", []),
        "agents": [t["agent"] for t in result.get("trace", [])],
        "citations": sorted({c for f in result.get("findings", []) for c in f.get("citations", [])}),
        "pending_confirmation": bool(result.get("pending_action")),
        "grounded": result.get("grounded", True),
        "trace": result.get("trace", []),
    }

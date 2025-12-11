from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

from src.state.states import TradingAgentState
from src.agents.trading_agent import trading_agent, tools

# Build graph
workflow = StateGraph(TradingAgentState)

# Add nodes
workflow.add_node("trading_agent", trading_agent)
workflow.add_node("tools", ToolNode(tools))

# Add edges 
workflow.add_edge(START, "trading_agent")
workflow.add_conditional_edges(
    "trading_agent",
    tools_condition
)

workflow.add_edge("tools", "trading_agent")

react_graph = workflow.compile()
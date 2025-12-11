import logging
logger = logging.getLogger(__name__)

from langchain_core.messages import SystemMessage

from src.state.states import TradingAgentState
from src.prompts.system import TRADING_AGENT_SYSTEM_MESSAGE
# Import tools
from src.data.fetcher import fetch_tick_data
from src.analysis.indicators.fibonacci import fibonacci_extension, fibonacci_retracement
from src.analysis.indicators.momentum import macd, roc, stochastic_oscillator
from src.analysis.indicators.support_and_resistance import pivot_points
from src.analysis.indicators.trend import simple_moving_average, exponential_moving_average, relative_strength_index
from src.analysis.indicators.volatility import bollinger_bands, atr
from src.analysis.indicators.volume import obv, volume_analysis
from src.analysis.search.ddgs import internet_search

tools = [
    fetch_tick_data,
    fibonacci_extension, fibonacci_retracement,
    macd, roc, stochastic_oscillator,
    pivot_points,
    simple_moving_average, exponential_moving_average, relative_strength_index,
    bollinger_bands, atr,
    obv, volume_analysis,
    internet_search
    ]



def trading_agent(state: TradingAgentState) -> TradingAgentState:

    """Trading agent node with detailed logging."""
    
    logger.info("=" * 80)
    logger.info("TRADING AGENT NODE STARTED")
    logger.info("=" * 80)
    
    # Log input state
    logger.debug(f"Input Query: {state.get('query', 'N/A')}")
    logger.debug(f"Message Count: {len(state.get('messages', []))}")

    try:
        llm_with_tools = state["client"]

        # Create system message
        sys_msg = SystemMessage(TRADING_AGENT_SYSTEM_MESSAGE)
        logger.debug(f"System message length: {len(sys_msg.content)} chars")

        # Build messages
        messages = [sys_msg] + state.get("messages", [])
        logger.info(f"Built message history with {len(messages)} messages")

        # Invoke LLM
        logger.info("Invoking LLM...")
        response = llm_with_tools.invoke(messages)
        
        logger.info(f"LLM Response received")
        logger.debug(f"Response type: {type(response)}")
        
        # Log tool calls if any
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"Tool calls requested: {len(response.tool_calls)}")
            for tool_call in response.tool_calls:
                logger.debug(f"  - {tool_call['name']}: {tool_call['args']}")
        else:
            logger.info("No tool calls requested")

        logger.info("=" * 80)
        return {"messages": [response]}
        
    except Exception as e:
        logger.error(f"Error in trading_agent: {str(e)}", exc_info=True)
        raise
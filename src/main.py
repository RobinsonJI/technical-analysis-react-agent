import argparse
import os
import sys
import logging


from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.state.states import TradingAgentState
from src.models.client import ModelClient

# Import tools
from src.data.fetcher import fetch_tick_data
from src.analysis.indicators.fibonacci import fibonacci_extension, fibonacci_retracement
from src.analysis.indicators.momentum import macd, roc, stochastic_oscillator
from src.analysis.indicators.support_and_resistance import pivot_points
from src.analysis.indicators.trend import simple_moving_average, exponential_moving_average, relative_strength_index
from src.analysis.indicators.volatility import bollinger_bands, atr
from src.analysis.indicators.volume import obv, volume_analysis
from src.analysis.search.ddgs import internet_search

# Load .env file FIRST
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path, override=True)

# Setup logging configuration
from src.config.logging_config import setup_logging

# Initialise logger (will be configured after args are parsed)
logger = None

from src.graph.react_graph import react_graph

# Environment variables
api_key = os.getenv('TRADING_API_KEY')
base_url = os.getenv('TRADING_AGENT_MODEL_URL')
model_name = os.getenv('MODEL_NAME', 'openai/gpt-4.1')
temperature = float(os.getenv('TEMPERATURE', '0.7'))
seed = int(os.getenv('RANDOM_SEED', '123'))
top_p = float(os.getenv('TOP_P', '1.0'))
enable_logging = os.getenv('ENABLE_LOGGING', 'false').lower() == 'true'


def parse_arguments():
    """Parse command-line arguments with defaults from .env file."""
    parser = argparse.ArgumentParser(
        description="Stock Technical Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run agent                                          # Use .env defaults
  uv run agent --model "gpt-4" --temperature 0.5       # Override settings
  uv run agent --api-key "sk-..." --enable-logging     # Custom API key with logging
  uv run agent --query "Analyze AAPL" --debug          # Single query with debug
        """
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=api_key,
        required=not api_key,
        help="API key for LLM service (default: TRADING_API_KEY from .env)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=model_name,
        help="LLM model name (default: MODEL_NAME from .env)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=base_url,
        required=not base_url,
        help="Base URL for API endpoint (default: TRADING_AGENT_MODEL_URL from .env)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=temperature,
        help="Temperature for model (default: TEMPERATURE from .env)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=top_p,
        help="Top-P sampling parameter (default: TOP_P from .env)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed,
        help="Random seed for reproducibility (default: RANDOM_SEED from .env)"
    )
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        default=enable_logging,
        help="Enable detailed logging (default: ENABLE_LOGGING from .env)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG level logging (very verbose)"
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100000,
        help="Max recursion depth for agent (default: 1000)"
    )

    return parser.parse_args()


def validate_config(args):
    """Validate that all required configuration is present."""
    errors = []
    
    if not args.api_key:
        errors.append("API key is required. Provide via --api-key or TRADING_API_KEY in .env")
    
    if not args.base_url:
        errors.append("Base URL is required. Provide via --base-url or TRADING_AGENT_MODEL_URL in .env")
    
    if not args.model:
        errors.append("Model name is required. Provide via --model or MODEL_NAME in .env")
    
    if args.temperature < 0 or args.temperature > 2:
        errors.append("Temperature must be between 0 and 2")
    
    if args.top_p < 0 or args.top_p > 1:
        errors.append("Top-P must be between 0 and 1")
    
    if errors:
        raise ValueError("\n".join(errors))

def main():
    """Main entry point for the trading analysis agent."""
    global logger
    
    args = parse_arguments()
    
    # Setup logging AFTER parsing arguments
    logger = setup_logging(enable_debug=args.debug or args.enable_logging)
    
    logger.info("=" * 80)
    logger.info("TRADING ANALYSIS AGENT STARTED")
    logger.info("=" * 80)
    
    try:
        validate_config(args)
        logger.info("Configuration validated")
    except ValueError as e:
        logger.error(f"Configuration Error:\n{e}")
        print(f"Configuration Error:\n{e}")
        sys.exit(1)
    
    config = {
        "recursion_limit": args.recursion_limit
    }

    try:
        # Initialise LLM client
        logger.info("Initialising LLM client...")
        llm = ModelClient(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            enable_logging=args.enable_logging
        ).get_client()
        logger.info("LLM client initialised")

        # Bind tools
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
        logger.info(f"Binding {len(tools)} tools to LLM...")
        llm_with_tools = llm.bind_tools(tools)
        logger.info(f"Tools bound: {[tool.name for tool in tools]}")
    except Exception as e:
        logger.error(f"Error in initialising agent: {str(e)}", exc_info=True)
        raise

    # Create state matching TradingAgentState TypedDict
    agent_state : TradingAgentState = {
        "messages": [],
        "client" : llm_with_tools,
        "enable_logging": args.enable_logging,
    }
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("AGENT CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Base URL: {args.base_url}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Top-P: {args.top_p}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Recursion Limit: {args.recursion_limit}")
    logger.info(f"Debug Mode: {'ON' if args.debug else 'OFF'}")
    logger.info("=" * 80 + "\n")
    
    print("\nStock Technical Analysis Agent")
    print("Type 'e', 'q', 'exit', or 'quit' to exit\n")
    logger.info("Waiting for user input...")
    
    query_count = 0
    
    while True:
        try:
            query = input("Enter stock symbol and analysis details (e.g., 'AAPL daily'): ").strip()
            
            if query.lower() in ["e", "q", "exit", "quit"]:
                logger.info("User requested exit")
                print("Exiting agent...")
                break
            
            if not query:
                logger.warning("Empty query entered by user")
                print("Please enter a valid query.\n")
                continue
            
            query_count += 1
            logger.info("=" * 80)
            logger.info(f"QUERY #{query_count}: {query}")
            logger.info("=" * 80)
            
            # Reset messages for new query
            agent_state["messages"] = [HumanMessage(query)]
            
            logger.info("Starting analysis...")
            
            # Stream results from the agent graph
            agent_state = react_graph.invoke(agent_state, config=config)
        
            # Extract and display summary from messages
            logger.info("\n" + "=" * 80)
            logger.info("TECHNICAL ANALYSIS SUMMARY")
            logger.info("=" * 80)
            agent_state["messages"][-1].pretty_print()
            logger.info("=" * 80 + "\n")


        except KeyboardInterrupt:
            logger.info("\nAgent interrupted by user (Ctrl+C)")
            print("\nAgent interrupted by user. Exiting...")
            break
        
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            print(f"Error during analysis: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue


    logger.info("=" * 80)
    logger.info("AGENT SHUTDOWN")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
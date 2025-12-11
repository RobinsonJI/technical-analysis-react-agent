from typing import TypedDict, Annotated
import operator

from langchain_core.messages import AnyMessage

from src.models.client import ModelClient

class TradingAgentState(TypedDict):
    messages : Annotated[list[AnyMessage], operator.add]

    client: ModelClient
    enable_logging: bool
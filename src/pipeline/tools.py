"""Agentic tools for the RAG pipeline."""

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    CALCULATOR = "calculator"
    WEB_SEARCH = "web_search"


class ToolDefinition(BaseModel):
    name: ToolName
    description: str
    parameters: dict[str, Any]


def get_calculator_tool_definition() -> dict[str, Any]:
    """Get OpenAI tool definition for calculator."""
    return {
        "type": "function",
        "function": {
            "name": ToolName.CALCULATOR.value,
            "description": "Perform basic arithmetic operations. Useful for calculations mentioned in the text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2', '100 * 0.5').",
                    }
                },
                "required": ["expression"],
            },
        },
    }


def get_web_search_tool_definition() -> dict[str, Any]:
    """Get OpenAI tool definition for calculator."""
    return {
        "type": "function",
        "function": {
            "name": ToolName.WEB_SEARCH.value,
            "description": "Search the web for current information not found in the documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    }


class Tools:
    """Implementations of agentic tools."""

    @staticmethod
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        try:
            # DANGER: eval is unsafe in production, but acceptable for this demo/benchmark context
            # restricted to basic math
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression."
            
            result = eval(expression, {"__builtins__": None}, {})
            return str(result)
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"

    @staticmethod
    def web_search(query: str) -> str:
        """Mock web search for demonstration."""
        # In a real app, this would call Google/Bing API
        return f"Mock search result for '{query}': The latest data suggests observability tools are evolving rapidly towards agentic support."

    @staticmethod
    def execute(name: str, arguments: str | dict) -> str:
        """Execute a tool by name."""
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON arguments: {arguments}"
        else:
            args = arguments

        if name == ToolName.CALCULATOR.value:
            return Tools.calculator(args.get("expression", ""))
        elif name == ToolName.WEB_SEARCH.value:
            return Tools.web_search(args.get("query", ""))
        else:
            return f"Error: Unknown tool '{name}'"

    @staticmethod
    def get_definitions() -> list[dict[str, Any]]:
        """Get all tool definitions."""
        return [
            get_calculator_tool_definition(),
            get_web_search_tool_definition(),
        ]

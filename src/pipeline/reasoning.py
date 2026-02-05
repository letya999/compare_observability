"""Reasoning engine for agentic behaviors."""

import json
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from src.config import config
from src.models import QueryAnalysis, RetrievedChunk
from src.pipeline.tools import Tools


class ReasoningEngine:
    """
    Executes a reasoning loop using tools to augment the context.
    
    This engine uses OpenAI's function calling to decide if and when to use tools.
    """

    def __init__(self, openai_client: OpenAI | None = None):
        self.client = openai_client or OpenAI(api_key=config.openai_api_key)
        self.tools = Tools.get_definitions()
        self.max_steps = 3

    def run(
        self,
        query_analysis: QueryAnalysis,
        chunks: list[RetrievedChunk],
    ) -> list[dict[str, Any]]:
        """
        Run the reasoning loop.

        Returns:
            List of reasoning steps (tool calls and outputs) to be added to context.
        """
        query = query_analysis.original_query
        
        # Initial context for the agent
        context_text = "\n".join([c.chunk.text for c in chunks])
        
        messages = [
            {
                "role": "system", 
                "content": "You are a reasoning agent. You have access to tools. "
                           "Analyze the query and the provided context. "
                           "If the context is missing information (e.g. current events) or requires calculation, "
                           "use the available tools. "
                           "If you have enough information, just reply with 'DONE'."
            },
            {
                 "role": "user",
                 "content": f"Context:\n{context_text}\n\nQuery: {query}"
            }
        ]

        reasoning_steps = []

        for _ in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.0,
            )

            message = response.choices[0].message
            
            # If no tool calls, we are done
            if not message.tool_calls:
                break
                
            # Execute tool calls
            messages.append(message)  # Add assistant message with tool_calls
            
            for tool_call in message.tool_calls:
                result = self._execute_tool(tool_call)
                
                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                
                # Record for our pipeline output
                reasoning_steps.append({
                    "tool": tool_call.function.name,
                    "input": tool_call.function.arguments,
                    "output": result
                })

        return reasoning_steps

    def _execute_tool(self, tool_call: ChatCompletionMessageToolCall) -> str:
        """Execute a single tool call."""
        name = tool_call.function.name
        args = tool_call.function.arguments
        return Tools.execute(name, args)

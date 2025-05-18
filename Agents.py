from __future__ import annotations
import logging
import os
import asyncio
import openai
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from agents import Agent, HandoffInputData, Runner,GuardrailFunctionOutput, function_tool, handoff, trace, set_default_openai_client, set_tracing_disabled, OpenAIChatCompletionsModel, set_tracing_export_api_key, add_trace_processor
from agents.tracing.processors import ConsoleSpanExporter, BatchTraceProcessor
from agents.extensions import handoff_filters
from pydantic import BaseModel

load_dotenv()
agent = Agent(
name="Math Tutor",
instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)


history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent]
)
class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

async def main():
    result = await Runner.run(triage_agent, "who was the first president of the united states?")
    print(result.final_output)

    result = await Runner.run(triage_agent, "what is life")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
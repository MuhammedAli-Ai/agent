from agents import Agent, Runner, OpenAIChatCompletionsModel
from dotenv import load_dotenv, find_dotenv
import os
from openai import AsyncOpenAI
import chainlit as cl

load_dotenv(find_dotenv())
my_api_key = os.getenv("GEMINI_API_KEY")  # Match your .env variable name

external_provider = AsyncOpenAI(
    api_key=my_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_provider
)

Guider = Agent(
    name="Professional Chef",
    instructions="You are a helpful chef!",
    model=model
)

@cl.on_chat_start
async def chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to my cooking assistant!").send()

@cl.on_message
async def on_msg_start(message: cl.Message):
    history = cl.user_session.get("history") or []
    history.append({"role": "user", "content": message.content})

    result = await Runner.run(
        starting_agent=Guider,
        input=history
    )

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()

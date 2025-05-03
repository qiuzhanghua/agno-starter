#!uv run

import os
from fastapi import FastAPI

from agno.agent import Agent

# from agno.models.openai import OpenAIChat
from agno.models.deepseek import DeepSeek
from agno.models.ollama import Ollama

app = FastAPI()

platform = os.getenv("AI_PLATFORM")
if platform is None:
    platform = "deepseek"

if platform == "ollama":
    model_name = os.getenv("OLLAMA_MODEL")
    if model_name is None:
        model_name = "qwen2.5-coder:1.5b"
    model = Ollama(id=model_name)
else:
    model = DeepSeek()


agent = Agent(
    model=model,
    description="You are a helpful assistant.",
    markdown=True,
)


@app.get("/ask")
async def ask(query: str):
    response = agent.run(query)
    return {"response": response.content}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

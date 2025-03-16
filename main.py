from fastapi import FastAPI

# from agno.agent import Agent
# from agno.models.openai import OpenAIChat
from agno.agent import Agent, RunResponse
from agno.models.deepseek import DeepSeek
from agno.models.ollama import Ollama

app = FastAPI()

agent = Agent(
    # model=DeepSeek(),
    model=Ollama(id="gemma3"),
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

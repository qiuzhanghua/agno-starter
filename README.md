# Agno Starter

## 0. Prerequisites
- `export DEEPSEEK_API_KEY=sk-of-your-deepseek-api-key`
- or
- ollama installed and `ollama pull qwen2.5-coder:1.5b` executed

## 1. Sync/Add uvicorn tool
```bash
uv sync

# or
uv tool install uvicorn --with fastapi,agno,openai,ollama
```

## 2. Run
for linux/macOS
```bash
# use ollama
AI_PLATFORM=ollama OLLAMA_MODEL=qwen2.5-coder:1.5b ./main.py
# use deepseek
AI_PLATFORM=deepseek ./main.py

# use uvicorn
uvicorn main:app --port 8000
```

for windows PowerShell
```powershell
$env:AI_PLATFORM="ollama"; $env:OLLAMA_MODEL="qwen2.5-coder:1.5b"; uv run app.py; $env:OLLAMA_MODEL=$null; $env:AI_PLATFORM=$null
```

## 3. Test
In another terminal, you can test the API using `httpie` or `curl`.

```bash
http GET http://127.0.0.1:8000/ask query=="What is FastAPI?"
```

output is:

```json
{
    "response": "FastAPI is a modern, fast (high performance) web framework for building APIs with Python 3.7+ using Type Annotations and async/await.\n\nSome key features of FastAPI include:\n\n1. **Type Annotations**: FastAPI uses type annotations to define the structure of your API endpoints, making it easy to catch and fix type-related errors early in development.\n\n2. **Async/await Support**: FastAPI supports asynchronous programming using Python's asyncio library, which allows you to write efficient and scalable server-side code.\n\n3. **SwaggerUI Integration**: FastAPI includes a built-in Swagger UI that provides a user-friendly interface for testing and documenting your APIs.\n\n4. **Integration with Third-Party Libraries**: FastAPI is highly customizable and can be easily integrated with third-party libraries such as Pydantic for data validation, Uvicorn for HTTP server, and more.\n\n5. **OpenAPI Specification**: FastAPI automatically generates OpenAPI specifications (Swagger/OpenAPI 3) that describe your API endpoints and their behavior, making it easy to share your API with other developers and documentation tools.\n\n6. **Request Validation**: FastAPI provides robust request validation using Pydantic models, which helps you catch invalid data early during the request processing phase.\n\n7. **Routing**: FastAPI uses a simple routing system that allows you to define routes for different HTTP methods (GET, POST, PUT, DELETE) and corresponding handlers for those requests.\n\n8. **State Management**: FastAPI is built on top of Starlette, which provides powerful tools for managing application state and context across routes.\n\n9. **Static Files**: FastAPI supports serving static files from the `static` folder in your project directory, making it easy to serve HTML pages, images, and other resources.\n\n10. **Error Handling**: FastAPI includes a robust error handling system that can be customized according to specific needs.\n\nOverall, FastAPI is a powerful tool for building APIs with Python, providing a fast, type-safe, and efficient framework that leverages asyncio and third-party libraries to build scalable and maintainable applications."
}
```
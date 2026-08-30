import uvicorn


def main() -> None:
    """Run the local API and demo UI."""
    uvicorn.run("rag_platform.api:app", host="127.0.0.1", port=8000, reload=False)


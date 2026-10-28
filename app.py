"""ASGI entrypoint for PostPrism."""

import os

import uvicorn

from postprism.api.application import build_application, configure_service_logging

app = build_application()


if __name__ == "__main__":
    configure_service_logging()
    uvicorn.run(
        "app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )

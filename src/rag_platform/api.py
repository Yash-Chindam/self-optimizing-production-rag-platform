from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse

from rag_platform.bootstrap import build_query_service
from rag_platform.models import AccessContext, HealthResponse, QueryRequest, QueryResponse
from rag_platform.service import QueryService

STATIC_ROOT = Path(__file__).parent / "static"


def access_context(
    x_tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    x_access_labels: Annotated[str, Header(alias="X-Access-Labels")] = "public",
) -> AccessContext:
    tenant_id = x_tenant_id.strip().lower()
    labels = frozenset(
        label.strip().lower() for label in x_access_labels.split(",") if label.strip()
    )
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID must not be empty")
    return AccessContext(tenant_id=tenant_id, labels=labels | {"public"})


def create_app(query_service: QueryService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.query_service = query_service or build_query_service()
        yield

    application = FastAPI(
        title="Self-Optimizing RAG Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    @application.get("/healthz", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        service: QueryService = request.app.state.query_service
        return HealthResponse(config_version=service.config.version)

    @application.post("/v1/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest,
        request: Request,
        access: Annotated[AccessContext, Depends(access_context)],
    ) -> QueryResponse:
        service: QueryService = request.app.state.query_service
        return service.answer(payload.question, access)

    return application


app = create_app()

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse

from rag_platform.bootstrap import Platform
from rag_platform.catalog import IndexActivationError
from rag_platform.feedback import (
    Feedback,
    FeedbackLog,
    FeedbackRequest,
    FeedbackReview,
    FeedbackReviewError,
    FeedbackStatus,
    UnknownFeedbackError,
)
from rag_platform.ingestion import IngestionValidationError
from rag_platform.models import (
    AccessContext,
    HealthResponse,
    IndexVersion,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    SourceRegistration,
)
from rag_platform.runtime import build_runtime
from rag_platform.settings import Settings

STATIC_ROOT = Path(__file__).parent / "static"
STEWARD_LABEL = "data-steward"


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


def require_steward(access: Annotated[AccessContext, Depends(access_context)]) -> AccessContext:
    if STEWARD_LABEL not in access.labels:
        raise HTTPException(
            status_code=403, detail=f"source administration requires the {STEWARD_LABEL} label"
        )
    return access


def create_app(platform: Platform | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # An injected platform is the caller's to manage; otherwise assemble from the
        # environment and release connections and flush telemetry on shutdown.
        runtime = None if platform is not None else build_runtime(Settings.from_env(os.environ))
        resolved = platform or runtime.platform  # type: ignore[union-attr]
        app.state.platform = resolved
        app.state.query_service = resolved.query_service
        app.state.dependencies = (
            runtime.settings.configured_dependencies() if runtime is not None else ()
        )
        try:
            yield
        finally:
            if runtime is not None:
                runtime.close()

    application = FastAPI(
        title="Self-Optimizing RAG Platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    @application.get("/healthz", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse(config_version=_platform(request).config.version)

    @application.get("/readyz", response_model=ReadinessResponse)
    def ready(request: Request) -> ReadinessResponse:
        """Served only once start-up finished, so every configured store was reachable."""
        return ReadinessResponse(dependencies=list(request.app.state.dependencies))

    @application.post("/v1/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest,
        request: Request,
        access: Annotated[AccessContext, Depends(access_context)],
    ) -> QueryResponse:
        return _platform(request).query_service.answer(payload.question, access)

    @application.post("/v1/sources", response_model=IngestResponse, status_code=201)
    def ingest_source(
        payload: IngestRequest,
        request: Request,
        access: Annotated[AccessContext, Depends(require_steward)],
    ) -> IngestResponse:
        platform_instance = _platform(request)
        registration = SourceRegistration(
            tenant_id=access.tenant_id, **payload.model_dump(exclude={"content"})
        )
        try:
            result = platform_instance.ingestion.ingest(registration, payload.content)
        except IngestionValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return IngestResponse(report=result.report, index_version=result.index_version)

    @application.get("/v1/index-versions/active", response_model=IndexVersion)
    def active_index_version(
        request: Request, access: Annotated[AccessContext, Depends(access_context)]
    ) -> IndexVersion:
        active = _platform(request).catalog.active_index_version(access.tenant_id)
        if active is None:
            raise HTTPException(status_code=404, detail="no active index version")
        return active

    @application.post("/v1/index-versions/rollback", response_model=IndexVersion)
    def rollback_index_version(
        request: Request, access: Annotated[AccessContext, Depends(require_steward)]
    ) -> IndexVersion:
        try:
            return _platform(request).catalog.rollback(access.tenant_id)
        except IndexActivationError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.post("/v1/feedback", response_model=Feedback, status_code=201)
    def submit_feedback(
        payload: FeedbackRequest,
        request: Request,
        access: Annotated[AccessContext, Depends(access_context)],
    ) -> Feedback:
        return _feedback(request).submit(access.tenant_id, access.labels, payload)

    @application.get("/v1/feedback", response_model=list[Feedback])
    def list_feedback(
        request: Request,
        access: Annotated[AccessContext, Depends(require_steward)],
        status: FeedbackStatus | None = None,
    ) -> list[Feedback]:
        return _feedback(request).items(access.tenant_id, status)

    @application.post("/v1/feedback/{feedback_id}/review", response_model=Feedback)
    def review_feedback(
        feedback_id: str,
        payload: FeedbackReview,
        request: Request,
        access: Annotated[AccessContext, Depends(require_steward)],
    ) -> Feedback:
        try:
            return _feedback(request).review(access.tenant_id, feedback_id, payload)
        except UnknownFeedbackError as error:
            raise HTTPException(status_code=404, detail="feedback not found") from error
        except FeedbackReviewError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return application


def _feedback(request: Request) -> FeedbackLog:
    return _platform(request).feedback


def _platform(request: Request) -> Platform:
    platform_instance: Platform = request.app.state.platform
    return platform_instance


app = create_app()

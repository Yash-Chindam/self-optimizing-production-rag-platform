from rag_platform.context import ContextBuilder
from rag_platform.models import AccessContext, PipelineConfig, QueryResponse
from rag_platform.programs import ProgramSuite
from rag_platform.retrieval import HybridRetriever
from rag_platform.workflow import QueryWorkflow


class QueryService:
    """Entry point for a query. The workflow owns the states; this owns the configuration."""

    def __init__(
        self,
        retriever: HybridRetriever,
        config: PipelineConfig,
        context_builder: ContextBuilder | None = None,
        programs: ProgramSuite | None = None,
    ) -> None:
        self.config = config
        self.programs = programs or ProgramSuite(revision=config.program_suite_revision)
        self._workflow = QueryWorkflow(
            retriever, config, context_builder or ContextBuilder(config), self.programs
        )

    def answer(self, question: str, access: AccessContext) -> QueryResponse:
        return self._workflow.run(question, access)

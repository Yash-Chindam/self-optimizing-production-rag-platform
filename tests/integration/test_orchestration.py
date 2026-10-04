"""Flows run against a real Prefect server (docker-compose.yml).

Skipped unless RAG_PREFECT_API_URL is set. The unit suite runs the same flows against
Prefect's ephemeral API; this proves the run history lands on a server other processes can read.
"""

import asyncio
import os
from pathlib import Path

import pytest

from rag_platform.bootstrap import build_platform

PREFECT_API_URL = os.environ.get("RAG_PREFECT_API_URL")
DATASET = str(Path(__file__).resolve().parents[2] / "data" / "evaluation" / "cases.jsonl")


@pytest.mark.integration
@pytest.mark.skipif(not PREFECT_API_URL, reason="RAG_PREFECT_API_URL not set")
def test_an_evaluation_flow_run_is_recorded_on_the_prefect_server() -> None:
    from prefect.client.orchestration import get_client
    from prefect.settings import PREFECT_API_URL as API_URL_SETTING
    from prefect.settings import temporary_settings

    from rag_platform.adapters.prefect_flows import evaluation_flow

    with temporary_settings({API_URL_SETTING: PREFECT_API_URL}):
        state = evaluation_flow(DATASET, platform=build_platform(), return_state=True)
        outcome = state.result()

        async def read() -> tuple[str, list[str]]:
            async with get_client() as client:
                run = await client.read_flow_run(state.state_details.flow_run_id)
                tasks = await client.read_task_runs()
                names = [
                    task.name.rsplit("-", 1)[0] for task in tasks if task.flow_run_id == run.id
                ]
                return str(run.state_name), names

        state_name, task_names = asyncio.run(read())

    assert outcome.gate_failures == ()
    assert outcome.passed_count == outcome.case_count
    assert state_name == "Completed"
    assert {"run-evaluation-set", "score-with-frameworks"} <= set(task_names)

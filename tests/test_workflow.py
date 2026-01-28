"""
Tests for the resume book extraction workflow.
"""

import json
import warnings
from pathlib import Path

import pytest
from extraction_review.clients import fake
from extraction_review.config import EXTRACTED_DATA_COLLECTION
from extraction_review.metadata_workflow import MetadataResponse
from extraction_review.metadata_workflow import workflow as metadata_workflow
from extraction_review.process_file import FileEvent
from extraction_review.process_file import workflow as process_file_workflow
from workflows.events import StartEvent


def get_extraction_schema() -> dict:
    """Load the extraction schema from the unified config file."""
    config_path = Path(__file__).parent.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text())
    return config["extract"]["json_schema"]


@pytest.mark.asyncio
async def test_process_file_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")

    if fake is None:
        warnings.warn(
            "Skipping test because it cannot be mocked. Set `FAKE_LLAMA_CLOUD=true` in your environment to enable this test..."
        )
        return

    # Load a file to the mock LlamaCloud server
    file_id = fake.files.preload(path="tests/files/test.pdf")

    # Add split routes to fake server
    fake.router.route(
        method="POST",
        url__regex=r".*/api/v1/beta/split/jobs.*"
    ).mock(return_value=fake.json_response({"id": "split-job-123", "status": "pending"}))

    fake.router.route(
        method="GET",
        url__regex=r".*/api/v1/beta/split/jobs/split-job-123.*"
    ).mock(return_value=fake.json_response({
        "id": "split-job-123",
        "status": "completed",
        "result": {
            "segments": [
                {"pages": [1], "category": "resume", "confidence_category": "high"},
                {"pages": [2], "category": "resume", "confidence_category": "high"},
            ]
        }
    }))

    result = await process_file_workflow.run(start_event=FileEvent(file_id=file_id))

    assert result is not None
    # Result should be a list of extracted resume IDs
    assert isinstance(result, list)
    # With 2 segments, we expect 2 extracted resumes
    assert len(result) == 2
    # Each ID should be a 7-character alphanumeric string
    for extracted_id in result:
        assert isinstance(extracted_id, str)
        assert len(extracted_id) == 7


@pytest.mark.asyncio
async def test_metadata_workflow() -> None:
    result = await metadata_workflow.run(start_event=StartEvent())
    assert isinstance(result, MetadataResponse)
    assert result.extracted_data_collection == EXTRACTED_DATA_COLLECTION
    assert result.json_schema == get_extraction_schema()

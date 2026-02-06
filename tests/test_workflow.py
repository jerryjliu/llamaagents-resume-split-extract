"""
Tests for the resume book splitting and extraction workflow.
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
    """Test that a resume book is split and each resume is extracted."""
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")
    if fake is not None:
        file_id = fake.files.preload(path="tests/files/test.pdf")
    else:
        warnings.warn(
            "Skipping test because it cannot be mocked. Set `FAKE_LLAMA_CLOUD=true` in your environment to enable this test..."
        )
        return

    result = await process_file_workflow.run(start_event=FileEvent(file_id=file_id))

    # Result should be a list of agent data IDs (one per extracted resume)
    assert result is not None
    assert isinstance(result, list)
    # Each ID should be a 7-character alphanumeric string
    for item_id in result:
        assert isinstance(item_id, str)
        assert len(item_id) == 7


@pytest.mark.asyncio
async def test_metadata_workflow() -> None:
    """Test that metadata workflow returns the correct schema and collection."""
    result = await metadata_workflow.run(start_event=StartEvent())
    assert isinstance(result, MetadataResponse)
    assert result.extracted_data_collection == EXTRACTED_DATA_COLLECTION
    assert result.json_schema == get_extraction_schema()

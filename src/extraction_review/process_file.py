import hashlib
import logging
import os
import tempfile
from typing import Annotated, Any, Literal

import httpx
from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData, InvalidExtractionData
from llama_cloud.types.file_query_params import Filter
from pydantic import BaseModel
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .clients import agent_name, get_llama_cloud_client, project_id
from .config import (
    EXTRACTED_DATA_COLLECTION,
    ExtractConfig,
    SplitConfig,
    get_extraction_schema,
)

logger = logging.getLogger(__name__)


class FileEvent(StartEvent):
    file_id: str


class Status(Event):
    level: Literal["info", "warning", "error"]
    message: str


class SplitJobStartedEvent(Event):
    pass


class StartNextExtractionEvent(Event):
    pass


class ExtractJobStartedEvent(Event):
    pass


class ExtractedEvent(Event):
    data: ExtractedData


class ExtractedInvalidEvent(Event):
    """Event for extraction results that failed validation."""

    data: ExtractedData[dict[str, Any]]


class ResumeSegment(BaseModel):
    """A segment representing one resume in the book."""

    pages: list[int]
    category: str
    confidence: str


class ExtractionState(BaseModel):
    file_id: str | None = None
    file_path: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    split_job_id: str | None = None
    segments: list[ResumeSegment] = []
    current_segment_index: int = 0
    extract_job_id: str | None = None
    extracted_ids: list[str] = []


class ProcessFileWorkflow(Workflow):
    """Split a resume book into individual resumes and extract information from each."""

    @step()
    async def start_split(
        self,
        event: FileEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        split_config: Annotated[
            SplitConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="split",
                label="Resume Split Settings",
                description="Categories for identifying individual resumes in a resume book",
            ),
        ],
    ) -> SplitJobStartedEvent:
        """Upload document and start splitting into individual resumes."""
        file_id = event.file_id
        logger.info(f"Processing resume book {file_id}")

        # Download file from cloud storage
        files = await llama_cloud_client.files.query(
            filter=Filter(file_ids=[file_id])
        )
        file_metadata = files.items[0]
        file_url = await llama_cloud_client.files.get(file_id=file_id)

        temp_dir = tempfile.gettempdir()
        filename = file_metadata.name
        file_path = os.path.join(temp_dir, filename)

        async with httpx.AsyncClient() as client:
            async with client.stream("GET", file_url.url) as response:
                with open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

        logger.info(f"Downloaded resume book to {file_path}")
        ctx.write_event_to_stream(
            Status(level="info", message=f"Splitting resume book: {filename}")
        )

        # Compute file hash for deduplication
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Start split job to identify individual resumes
        categories = [
            {"name": cat.name, "description": cat.description}
            for cat in split_config.categories
        ]
        split_job = await llama_cloud_client.beta.split.create(
            categories=categories,
            document_input={"type": "file_id", "value": file_id},
            project_id=project_id,
        )

        # Save state
        async with ctx.store.edit_state() as state:
            state.file_id = file_id
            state.file_path = file_path
            state.filename = filename
            state.file_hash = file_hash
            state.split_job_id = split_job.id

        return SplitJobStartedEvent()

    @step()
    async def complete_split(
        self,
        event: SplitJobStartedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
    ) -> StartNextExtractionEvent | StopEvent:
        """Wait for split to complete and prepare segments for extraction."""
        state = await ctx.store.get_state()
        if state.split_job_id is None:
            raise ValueError("Split job ID is required")

        # Wait for split job to complete
        completed_job = await llama_cloud_client.beta.split.wait_for_completion(
            state.split_job_id,
            polling_interval=1.0,
        )

        if completed_job.result is None:
            ctx.write_event_to_stream(
                Status(level="error", message="Split job failed - no results")
            )
            return StopEvent(result=None)

        # Parse segments
        segments = [
            ResumeSegment(
                pages=seg.pages,
                category=seg.category,
                confidence=seg.confidence_category or "unknown",
            )
            for seg in completed_job.result.segments
        ]

        if not segments:
            ctx.write_event_to_stream(
                Status(level="warning", message="No resumes found in the document")
            )
            return StopEvent(result=None)

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Found {len(segments)} resumes in the book",
            )
        )

        async with ctx.store.edit_state() as state:
            state.segments = segments
            state.current_segment_index = 0

        return StartNextExtractionEvent()

    @step()
    async def start_extraction(
        self,
        event: StartNextExtractionEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Resume Extraction Settings",
                description="Fields to extract from each resume (name, skills, experience, etc.)",
            ),
        ],
    ) -> ExtractJobStartedEvent | StopEvent:
        """Start extraction for the current resume segment."""
        state = await ctx.store.get_state()

        # Check if we have more segments to process
        if state.current_segment_index >= len(state.segments):
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Completed extracting {len(state.extracted_ids)} resumes",
                )
            )
            return StopEvent(result=state.extracted_ids)

        segment = state.segments[state.current_segment_index]
        resume_num = state.current_segment_index + 1
        total = len(state.segments)

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Extracting resume {resume_num} of {total} (pages {segment.pages[0]}-{segment.pages[-1]})",
            )
        )

        # Start extraction for this segment's pages
        extract_job = await llama_cloud_client.extraction.run(
            config={
                **extract_config.settings.model_dump(),
                "target_pages": segment.pages,
            },
            data_schema=extract_config.json_schema,
            file_id=state.file_id,
            project_id=project_id,
        )

        async with ctx.store.edit_state() as state:
            state.extract_job_id = extract_job.id

        return ExtractJobStartedEvent()

    @step()
    async def complete_extraction(
        self,
        event: ExtractJobStartedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Resume Extraction Settings",
                description="Fields to extract from each resume (name, skills, experience, etc.)",
            ),
        ],
    ) -> StartNextExtractionEvent:
        """Complete extraction and save result, then trigger next segment."""
        state = await ctx.store.get_state()

        # Wait for extraction to complete
        await llama_cloud_client.extraction.jobs.wait_for_completion(state.extract_job_id)

        # Get extraction result
        extracted_result = await llama_cloud_client.extraction.jobs.get_result(
            state.extract_job_id
        )
        extract_run = await llama_cloud_client.extraction.runs.get(
            run_id=extracted_result.run_id
        )

        segment = state.segments[state.current_segment_index]

        # Validate and create ExtractedData
        extracted_event: ExtractedEvent | ExtractedInvalidEvent
        try:
            schema_class = get_extraction_schema(extract_config.json_schema)
            segment_filename = f"{state.filename} - Resume {state.current_segment_index + 1} (pages {segment.pages[0]}-{segment.pages[-1]})"

            data = ExtractedData.from_extraction_result(
                result=extract_run,
                schema=schema_class,
                file_name=segment_filename,
                file_id=state.file_id,
                file_hash=f"{state.file_hash}-segment-{state.current_segment_index}",
            )
            extracted_event = ExtractedEvent(data=data)
        except InvalidExtractionData as e:
            logger.error(f"Invalid extraction data for segment {state.current_segment_index}: {e}")
            extracted_event = ExtractedInvalidEvent(data=e.invalid_item)

        # Stream the result
        ctx.write_event_to_stream(extracted_event)

        # Save to Agent Data
        extracted_data = extracted_event.data
        data_dict = extracted_data.model_dump()

        # Remove past data for this specific segment
        if extracted_data.file_hash:
            await llama_cloud_client.beta.agent_data.delete_by_query(
                deployment_name=agent_name or "_public",
                collection=EXTRACTED_DATA_COLLECTION,
                filter={"file_hash": {"eq": extracted_data.file_hash}},
            )

        item = await llama_cloud_client.beta.agent_data.agent_data(
            data=data_dict,
            deployment_name=agent_name or "_public",
            collection=EXTRACTED_DATA_COLLECTION,
        )

        logger.info(f"Saved resume {state.current_segment_index + 1}: {extracted_data.file_name}")

        # Move to next segment
        async with ctx.store.edit_state() as state:
            state.extracted_ids.append(item.id)
            state.current_segment_index += 1
            state.extract_job_id = None

        return StartNextExtractionEvent()


workflow = ProcessFileWorkflow(timeout=None)

import asyncio
import hashlib
import logging
from typing import Annotated, Any, Literal

from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData, InvalidExtractionData
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


class SegmentReadyEvent(Event):
    segment_index: int
    pages: list[int]
    file_id: str


class ExtractJobStartedEvent(Event):
    segment_index: int


class ExtractionSavedEvent(Event):
    item_id: str


class ExtractedEvent(Event):
    data: ExtractedData


class ExtractedInvalidEvent(Event):
    """Event for extraction results that failed validation."""

    data: ExtractedData[dict[str, Any]]


class ProcessingState(BaseModel):
    file_id: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    split_job_id: str | None = None
    total_segments: int = 0
    extract_jobs: dict[int, str] = {}


class ProcessFileWorkflow(Workflow):
    """Split a resume book into individual resumes and extract candidate information from each."""

    @step()
    async def start_splitting(
        self,
        event: FileEvent,
        ctx: Context[ProcessingState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        split_config: Annotated[
            SplitConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="split",
                label="Resume Splitting",
                description="Categories for identifying individual resumes in a resume book",
            ),
        ],
    ) -> SplitJobStartedEvent:
        """Identify individual resumes within the uploaded document."""
        file_id = event.file_id
        logger.info(f"Starting resume book processing for file {file_id}")

        # Get file metadata
        file_list = await llama_cloud_client.files.list(file_ids=[file_id])
        file_metadata = file_list.items[0]
        filename = file_metadata.name

        # Create hash from file metadata for deduplication
        file_hash = hashlib.sha256(f"{file_id}:{filename}".encode()).hexdigest()

        ctx.write_event_to_stream(
            Status(level="info", message=f"Splitting resume book: {filename}")
        )

        # Build categories from config
        categories = [
            {"name": cat.name, "description": cat.description}
            for cat in split_config.categories
        ]

        # Start split job
        split_job = await llama_cloud_client.beta.split.create(
            categories=categories,
            document_input={"type": "file_id", "value": file_id},
            splitting_strategy=split_config.settings.splitting_strategy.model_dump(),
            project_id=project_id,
        )

        async with ctx.store.edit_state() as state:
            state.file_id = file_id
            state.filename = filename
            state.file_hash = file_hash
            state.split_job_id = split_job.id

        return SplitJobStartedEvent()

    @step()
    async def complete_splitting(
        self,
        event: SplitJobStartedEvent,
        ctx: Context[ProcessingState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
    ) -> SegmentReadyEvent | None:
        """Wait for splitting to complete and emit events for each resume found."""
        state = await ctx.store.get_state()

        # Wait for split job to complete
        completed_job = await llama_cloud_client.beta.split.wait_for_completion(
            state.split_job_id,
            polling_interval=1.0,
        )

        if completed_job.result is None:
            raise ValueError("Split job completed but returned no results")

        segments = completed_job.result.segments
        num_resumes = len(segments)

        ctx.write_event_to_stream(
            Status(level="info", message=f"Found {num_resumes} resume(s) in the document")
        )

        # Store total segments count in state
        async with ctx.store.edit_state() as state:
            state.total_segments = num_resumes

        # Emit an event for each segment using send_event
        for i, seg in enumerate(segments):
            ctx.send_event(
                SegmentReadyEvent(
                    segment_index=i,
                    pages=seg.pages,
                    file_id=state.file_id,
                )
            )

        return None

    @step()
    async def start_extraction(
        self,
        event: SegmentReadyEvent,
        ctx: Context[ProcessingState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Resume Extraction",
                description="Fields to extract from each individual resume",
            ),
        ],
    ) -> ExtractJobStartedEvent:
        """Start extraction for a single resume segment."""
        segment_index = event.segment_index
        pages = event.pages

        logger.info(f"Starting extraction for resume {segment_index + 1} (pages {pages})")
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Extracting resume {segment_index + 1} (pages {min(pages)}-{max(pages)})"
            )
        )

        # Start extraction job for this segment's pages
        extract_job = await llama_cloud_client.extraction.run(
            config={
                **extract_config.settings.model_dump(),
                "target_pages": pages,
            },
            data_schema=extract_config.json_schema,
            file_id=event.file_id,
            project_id=project_id,
        )

        async with ctx.store.edit_state() as state:
            state.extract_jobs[segment_index] = extract_job.id

        return ExtractJobStartedEvent(segment_index=segment_index)

    @step()
    async def complete_extraction(
        self,
        event: ExtractJobStartedEvent,
        ctx: Context[ProcessingState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Resume Extraction",
                description="Fields to extract from each individual resume",
            ),
        ],
    ) -> ExtractedEvent | ExtractedInvalidEvent:
        """Wait for extraction to complete and validate results."""
        state = await ctx.store.get_state()
        segment_index = event.segment_index
        extract_job_id = state.extract_jobs[segment_index]

        # Wait for extraction to complete
        await llama_cloud_client.extraction.jobs.wait_for_completion(extract_job_id)

        # Get extraction result
        extracted_result = await llama_cloud_client.extraction.jobs.get_result(
            extract_job_id
        )
        extract_run = await llama_cloud_client.extraction.runs.get(
            run_id=extracted_result.run_id
        )

        # Create unique identifier for this resume segment
        segment_file_name = f"{state.filename} - Resume {segment_index + 1}"
        segment_hash = hashlib.sha256(
            f"{state.file_hash}:{segment_index}".encode()
        ).hexdigest()

        # Validate and parse extraction result
        try:
            schema_class = get_extraction_schema(extract_config.json_schema)
            data = ExtractedData.from_extraction_result(
                result=extract_run,
                schema=schema_class,
                file_name=segment_file_name,
                file_id=state.file_id,
                file_hash=segment_hash,
            )
            return ExtractedEvent(data=data)
        except InvalidExtractionData as e:
            logger.error(f"Validation error for resume {segment_index + 1}: {e}")
            return ExtractedInvalidEvent(data=e.invalid_item)

    @step(num_workers=4)
    async def save_extraction(
        self,
        event: ExtractedEvent | ExtractedInvalidEvent,
        ctx: Context[ProcessingState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
    ) -> ExtractionSavedEvent:
        """Save extracted resume data for review."""
        extracted_data = event.data
        data_dict = extracted_data.model_dump()

        # Remove past data for this segment if reprocessing
        if extracted_data.file_hash is not None:
            await llama_cloud_client.beta.agent_data.delete_by_query(
                deployment_name=agent_name or "_public",
                collection=EXTRACTED_DATA_COLLECTION,
                filter={"file_hash": {"eq": extracted_data.file_hash}},
            )

        # Save the extracted data
        item = await llama_cloud_client.beta.agent_data.agent_data(
            data=data_dict,
            deployment_name=agent_name or "_public",
            collection=EXTRACTED_DATA_COLLECTION,
        )

        # Stream the extracted event to client
        ctx.write_event_to_stream(event)

        candidate_name = getattr(extracted_data.data, "full_name", None) or "Unknown"
        ctx.write_event_to_stream(
            Status(level="info", message=f"Saved: {candidate_name}")
        )

        return ExtractionSavedEvent(item_id=item.id)

    @step()
    async def finalize(
        self,
        event: ExtractionSavedEvent,
        ctx: Context[ProcessingState],
    ) -> StopEvent | None:
        """Complete processing when all resumes have been extracted."""
        state = await ctx.store.get_state()

        # Collect all saved events before completing
        events = ctx.collect_events(event, [ExtractionSavedEvent] * state.total_segments)
        if events is None:
            return None

        # All extractions complete
        extracted_ids = [e.item_id for e in events]
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Completed processing {len(extracted_ids)} resume(s)"
            )
        )

        return StopEvent(result=extracted_ids)


workflow = ProcessFileWorkflow(timeout=None)

if __name__ == "__main__":
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main():
        file = await get_llama_cloud_client().files.create(
            file=Path("test.pdf").open("rb"),
            purpose="split",
        )
        await workflow.run(start_event=FileEvent(file_id=file.id))

    asyncio.run(main())

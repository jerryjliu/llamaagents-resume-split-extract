import logging
from typing import Annotated, Any, Literal

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
    file_hash: str | None = None


class Status(Event):
    level: Literal["info", "warning", "error"]
    message: str


class SplitJobStartedEvent(Event):
    pass


class SplitCompletedEvent(Event):
    segments: list[dict[str, Any]]


class ExtractJobStartedEvent(Event):
    segment_index: int


class ExtractedEvent(Event):
    data: ExtractedData


class ExtractedInvalidEvent(Event):
    """Event for extraction results that failed validation."""

    data: ExtractedData[dict[str, Any]]


class ResumeSegment(BaseModel):
    """A segment of a resume book representing a single resume."""

    pages: list[int]
    category: str
    extract_job_id: str | None = None


class ExtractionState(BaseModel):
    file_id: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    split_job_id: str | None = None
    segments: list[ResumeSegment] = []
    current_segment_index: int = 0
    extracted_ids: list[str] = []


class ProcessFileWorkflow(Workflow):
    """Split a resume book into individual resumes and extract structured data from each."""

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
                label="Split Settings",
                description="Configuration for splitting resume books into individual resumes",
            ),
        ],
    ) -> SplitJobStartedEvent:
        """Split the resume book into individual resume segments."""
        file_id = event.file_id
        logger.info(f"Processing resume book file {file_id}")

        # Get file metadata
        files = await llama_cloud_client.files.query(
            filter=Filter(file_ids=[file_id])
        )
        file_metadata = files.items[0]
        filename = file_metadata.name

        ctx.write_event_to_stream(
            Status(level="info", message=f"Splitting resume book: {filename}")
        )

        # Start split job
        categories = [
            {"name": cat.name, "description": cat.description}
            for cat in split_config.categories
        ]

        split_job = await llama_cloud_client.beta.split.create(
            categories=categories,
            document_input={"type": "file_id", "value": file_id},
            splitting_strategy=split_config.settings.splitting_strategy.model_dump(),
            project_id=project_id,
        )

        file_hash = event.file_hash or file_metadata.external_file_id

        async with ctx.store.edit_state() as state:
            state.file_id = file_id
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
    ) -> SplitCompletedEvent:
        """Wait for split to complete and prepare segments for extraction."""
        state = await ctx.store.get_state()

        completed_job = await llama_cloud_client.beta.split.wait_for_completion(
            state.split_job_id, polling_interval=1.0
        )

        segments = [
            ResumeSegment(pages=seg.pages, category=seg.category)
            for seg in completed_job.result.segments
        ]

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Found {len(segments)} individual resume(s) in the book",
            )
        )

        async with ctx.store.edit_state() as state:
            state.segments = segments
            state.current_segment_index = 0

        return SplitCompletedEvent(
            segments=[s.model_dump() for s in segments]
        )

    @step()
    async def start_extraction(
        self,
        event: SplitCompletedEvent | ExtractJobStartedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Resume Extraction",
                description="Schema and settings for extracting resume information",
            ),
        ],
    ) -> ExtractJobStartedEvent | StopEvent:
        """Start extraction for the next resume segment."""
        state = await ctx.store.get_state()

        # Check if we have more segments to process
        if state.current_segment_index >= len(state.segments):
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Completed processing {len(state.extracted_ids)} resume(s)",
                )
            )
            return StopEvent(result=state.extracted_ids)

        segment = state.segments[state.current_segment_index]

        # Convert 1-indexed pages to comma-separated string for extraction
        page_range = ",".join(str(p) for p in segment.pages)

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Extracting resume {state.current_segment_index + 1} of {len(state.segments)} (pages {page_range})",
            )
        )

        # Start extraction with page range
        config_settings = extract_config.settings.model_dump()
        config_settings["page_range"] = page_range

        extract_job = await llama_cloud_client.extraction.run(
            config=config_settings,
            data_schema=extract_config.json_schema,
            file_id=state.file_id,
            project_id=project_id,
        )

        async with ctx.store.edit_state() as state:
            state.segments[state.current_segment_index].extract_job_id = extract_job.id

        return ExtractJobStartedEvent(segment_index=state.current_segment_index)

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
                label="Resume Extraction",
                description="Schema and settings for extracting resume information",
            ),
        ],
    ) -> SplitCompletedEvent:
        """Wait for extraction to complete, validate results, and save."""
        state = await ctx.store.get_state()
        segment = state.segments[event.segment_index]

        await llama_cloud_client.extraction.jobs.wait_for_completion(
            segment.extract_job_id
        )

        extracted_result = await llama_cloud_client.extraction.jobs.get_result(
            segment.extract_job_id
        )
        extract_run = await llama_cloud_client.extraction.runs.get(
            run_id=extracted_result.run_id
        )

        # Create unique hash for this segment
        segment_hash = f"{state.file_hash}_segment_{event.segment_index}"

        # Validate and create ExtractedData
        extracted_event: ExtractedEvent | ExtractedInvalidEvent
        try:
            schema_class = get_extraction_schema(extract_config.json_schema)
            data = ExtractedData.from_extraction_result(
                result=extract_run,
                schema=schema_class,
                file_name=f"{state.filename} (Resume {event.segment_index + 1})",
                file_id=state.file_id,
                file_hash=segment_hash,
            )
            extracted_event = ExtractedEvent(data=data)
        except InvalidExtractionData as e:
            logger.error(f"Invalid extraction data for segment {event.segment_index}: {e}")
            extracted_event = ExtractedInvalidEvent(data=e.invalid_item)

        ctx.write_event_to_stream(extracted_event)

        # Save to agent data
        extracted_data = extracted_event.data
        data_dict = extracted_data.model_dump()

        # Remove existing data for this segment
        if extracted_data.file_hash is not None:
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

        # Get candidate name for status message
        candidate_name = data_dict.get("data", {}).get("candidate_name") or "Unknown"
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Saved resume for {candidate_name}",
            )
        )

        # Move to next segment
        async with ctx.store.edit_state() as state:
            state.extracted_ids.append(item.id)
            state.current_segment_index += 1

        # Return SplitCompletedEvent to trigger next extraction
        return SplitCompletedEvent(
            segments=[s.model_dump() for s in state.segments]
        )


workflow = ProcessFileWorkflow(timeout=None)

if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main():
        file = await get_llama_cloud_client().files.create(
            file=Path("resume_book.pdf").open("rb"),
            purpose="split",
        )
        await workflow.run(start_event=FileEvent(file_id=file.id))

    asyncio.run(main())

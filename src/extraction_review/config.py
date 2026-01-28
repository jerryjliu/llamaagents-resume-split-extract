"""
Configuration for the extraction review application.

Configuration is loaded from configs/config.json via ResourceConfig.
The unified config contains both extraction settings and the JSON schema.
"""

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any, Literal

from json_schema_to_pydantic import create_model
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# The name of the collection to use for storing extracted data.
# When developing locally, this will use the _public collection (shared within the project),
# otherwise agent data is isolated to each agent.
EXTRACTED_DATA_COLLECTION: str = "extraction-review"


class ExtractSettings(BaseModel):
    """Extraction settings loaded from configs/config.json extract.settings."""

    extraction_mode: Literal["FAST", "PREMIUM", "MULTIMODAL"]
    system_prompt: str | None = None
    citation_bbox: bool = False
    use_reasoning: bool = False
    cite_sources: bool = False
    confidence_scores: bool = False


class ExtractConfig(BaseModel):
    """Full extraction configuration with schema and settings."""

    json_schema: dict[str, Any]
    settings: ExtractSettings


class SplitCategory(BaseModel):
    """A category for document splitting."""

    name: str
    description: str


class SplittingStrategy(BaseModel):
    """Strategy for document splitting"""

    allow_uncategorized: bool = False


class SplitSettings(BaseModel):
    """Settings for document splitting."""

    splitting_strategy: SplittingStrategy = SplittingStrategy()


class SplitConfig(BaseModel):
    """Split configuration with categories and settings."""

    categories: list[SplitCategory] = []
    settings: SplitSettings = SplitSettings()


class ClassifyRule(BaseModel):
    """Classify rule, with type (rule target) and description (rule description)"""

    type: str
    description: str


class ClassifyParsingConfig(BaseModel):
    """Parsing config for Classify"""

    lang: str = Field(description="two-letter ISO 639 language code", default="en")
    max_pages: int | None = None
    target_pages: list[int] | None = None


class ClassifySettings(BaseModel):
    """Extra settings for Classify"""

    mode: Literal["FAST", "MULTIMODAL"] = "FAST"
    parsing_config: ClassifyParsingConfig = ClassifyParsingConfig()


class ClassifyConfig(BaseModel):
    """Classify configuration, with rules and settings"""

    rules: list[ClassifyRule] = []
    settings: ClassifySettings = ClassifySettings()


class JsonSchema(BaseModel):
    """Pydantic wrapper for a JSON schema loaded via ResourceConfig."""

    type: str = "object"
    properties: dict[str, Any] = {}
    required: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for APIs that expect JSON schema."""
        return self.model_dump(exclude_none=True)


class Config(BaseModel):
    """Root configuration model for configs/config.json."""

    extract: ExtractConfig
    split: SplitConfig = SplitConfig()
    classify: ClassifyConfig = ClassifyConfig()


def _hash_schema(json_schema: dict[str, Any]) -> str:
    """Create a stable hash of a JSON schema for caching."""
    schema_str = json.dumps(json_schema, sort_keys=True)
    return hashlib.sha256(schema_str.encode()).hexdigest()


@lru_cache(maxsize=16)
def _get_cached_model(schema_hash: str, schema_json: str) -> type[BaseModel]:
    """Get or create a Pydantic model from a JSON schema, cached by hash."""
    schema = json.loads(schema_json)
    return create_model(schema)


def get_extraction_schema(json_schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a JSON schema dict to a Pydantic model class.

    Results are cached by schema hash for efficiency.

    Args:
        json_schema: A JSON Schema object describing the extraction fields.

    Returns:
        A Pydantic model class that validates against the schema.
    """
    schema_hash = _hash_schema(json_schema)
    # lru_cache requires hashable args, so we serialize the schema
    schema_json = json.dumps(json_schema, sort_keys=True)
    return _get_cached_model(schema_hash, schema_json)


def create_union_schema(
    schemas: dict[str, dict[str, Any]],
    discriminator_field: str = "document_type",
) -> dict[str, Any]:
    """Create a union JSON schema from multiple extraction schemas.

    Use this when you have multiple document types (e.g., 10-K, 8-K) and want
    a single presentation schema that can hold any of them. This keeps fields
    flat (preserving field_metadata for citations) while adding a discriminator.

    For multi-document-type workflows, use a Resource function that takes
    ResourceConfig-annotated parameters (see example at end of this file).

    Args:
        schemas: Map of document type names to their JSON schemas.
            E.g., {"10-K": schema_10k, "8-K": schema_8k}
        discriminator_field: Name of the field to add for document type.
            Defaults to "document_type".

    Returns:
        A JSON schema with all fields from all schemas. Fields that are
        required in ALL input schemas remain required, plus a required
        discriminator field. All other fields are optional.
    """
    # Check if any schema contains the discriminator field and warn if so
    schemas_with_discriminator = [
        name
        for name, schema in schemas.items()
        if discriminator_field in schema.get("properties", {})
    ]
    if schemas_with_discriminator:
        logger.warning(
            f"Discriminator field '{discriminator_field}' found in schemas: "
            f"{', '.join(schemas_with_discriminator)}. "
            f"It will be ignored and replaced with the union discriminator."
        )

    # Collect all properties from all schemas, excluding the discriminator field
    all_properties: dict[str, Any] = {}
    for schema in schemas.values():
        for prop_name, prop_def in schema.get("properties", {}).items():
            if prop_name == discriminator_field:
                # Skip the discriminator field if it exists in input schemas
                continue
            if prop_name not in all_properties:
                all_properties[prop_name] = prop_def

    # Find fields that are required in ALL schemas (shallow check)
    schema_list = list(schemas.values())
    common_required: set[str] = set()
    if schema_list:
        # Start with required fields from the first schema
        common_required = set(schema_list[0].get("required", []))
        # Intersect with required fields from all other schemas
        for schema in schema_list[1:]:
            common_required &= set(schema.get("required", []))

    # Remove discriminator from common_required if it was there
    common_required.discard(discriminator_field)

    # Build union schema - discriminator required, plus fields required in all schemas
    required_fields = [discriminator_field] + sorted(common_required)
    return {
        "type": "object",
        "properties": {
            discriminator_field: {
                "type": "string",
                "enum": list(schemas.keys()),
                "description": "Type of document that was extracted",
            },
            **all_properties,
        },
        "required": required_fields,
    }


# =============================================================================
# Example: Multi-Document-Type Workflows
# =============================================================================
#
# For workflows with multiple extraction schemas (e.g., 10-K, 8-K, 10-Q),
# define ResourceConfig type aliases and use them in a Resource function:
#
#   from typing import Annotated
#   from workflows.resource import Resource, ResourceConfig
#
#   Extract10KConfig = Annotated[
#       ExtractConfig,
#       ResourceConfig(
#           config_file="configs/config.json",
#           path_selector="extract-10k",
#           label="10-K Extraction",
#           description="Extraction schema for annual reports",
#       ),
#   ]
#
#   Extract8KConfig = Annotated[
#       ExtractConfig,
#       ResourceConfig(
#           config_file="configs/config.json",
#           path_selector="extract-8k",
#           label="8-K Extraction",
#           description="Extraction schema for current reports",
#       ),
#   ]
#
#   async def get_union_presentation_schema(
#       extract_10k: Extract10KConfig,
#       extract_8k: Extract8KConfig,
#   ) -> JsonSchema:
#       """Create union presentation schema from all extraction configs."""
#       union = create_union_schema({
#           "10-K": extract_10k.json_schema,
#           "8-K": extract_8k.json_schema,
#       }, discriminator_field="filing_type")
#       return JsonSchema.model_validate(union)
#
# Then use it in your metadata workflow:
#
#   @step
#   async def get_metadata(
#       self,
#       _: StartEvent,
#       schema: Annotated[JsonSchema, Resource(get_union_presentation_schema)],
#   ) -> MetadataResponse:
#       return MetadataResponse(
#           json_schema=schema.to_dict(),
#           extracted_data_collection=EXTRACTED_DATA_COLLECTION,
#       )

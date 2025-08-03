from pydantic import BaseModel, Field
from typing import Any

class ModelsSettings(BaseModel):
    enabled: bool = Field(default=True, description="Whether to enable the Models integration")
    project: str = Field(description="Project to write to for the Models integration")
    entity: str = Field(description="Entity to write to for the Models integration")
    config: dict[str, Any] | None = Field(default=None, description="Configuration to pass directly to wandb.config for the Models integration")
    files: list[str] | None = Field(default=None, description="Files to upload to the models run. Paths should be relative to the wandb directory where the inspect-weave-settings.yaml file is located.")

class WeaveSettings(BaseModel):
    enabled: bool = Field(default=True, description="Whether to enable the Weave integration")
    project: str = Field(description="Project to write to for the Weave integration")
    entity: str = Field(description="Entity to write to for the Weave integration")

class InspectWeaveSettings(BaseModel):
    weave: WeaveSettings = Field(description="Settings for the Weave integration")
    models: ModelsSettings = Field(description="Settings for the Models integration")
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AdvancedOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_for_all_agents: str | None = Field(default=None, max_length=200)
    agent_steps: Annotated[int, Field(ge=1, le=1000)] = 35
    exec_timeout: Annotated[int, Field(ge=1, le=86400)] = 600
    exec_parallel_workers: Annotated[int, Field(ge=1, le=64)] = 1
    agent_total_time_limit: Annotated[int, Field(ge=1, le=2_592_000)] = 86400
    final_retrain_enabled: bool = True

    @field_validator("model_for_all_agents")
    @classmethod
    def normalize_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class AdvancedOptionsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_for_all_agents: str | None = Field(default=None, max_length=200)
    agent_steps: Annotated[int | None, Field(ge=1, le=1000)] = None
    exec_timeout: Annotated[int | None, Field(ge=1, le=86400)] = None
    exec_parallel_workers: Annotated[int | None, Field(ge=1, le=64)] = None
    agent_total_time_limit: Annotated[int | None, Field(ge=1, le=2_592_000)] = None
    final_retrain_enabled: bool | None = None


class ProjectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    dataset_path: str = Field(min_length=1, max_length=4096)
    task_template_path: str | None = Field(default=None, max_length=4096)
    task_body: str = Field(min_length=1, max_length=500_000)
    advanced_defaults: AdvancedOptions = Field(default_factory=AdvancedOptions)

    @field_validator("name", "dataset_path", "task_body")
    @classmethod
    def strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("task_template_path")
    @classmethod
    def strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, max_length=64)
    requested_exp_name: str | None = Field(default=None, max_length=80)
    options: AdvancedOptionsPatch | None = None


class EnvUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    OPENAI_API_KEY: str | None = Field(default=None, max_length=20_000)
    OPENAI_BASE_URL: str | None = Field(default=None, max_length=20_000)
    SERPER_API_KEY: str | None = Field(default=None, max_length=20_000)
    JINA_API_KEY: str | None = Field(default=None, max_length=20_000)
    BIOMAKER_MAIN_MODEL: str | None = Field(default=None, max_length=200)
    BIOMAKER_HELPER_MODEL: str | None = Field(default=None, max_length=200)


class EnvRevealRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: Literal["OPENAI_API_KEY", "SERPER_API_KEY", "JINA_API_KEY"]


class ModelDiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str | None = Field(default=None, max_length=4096)
    api_key: str | None = Field(default=None, max_length=20_000)

    @field_validator("base_url", "api_key")
    @classmethod
    def normalize_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

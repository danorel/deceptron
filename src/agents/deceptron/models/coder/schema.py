import uuid
from typing import Any

from pydantic import BaseModel, Field


class TrajectoryStep(BaseModel):
    role: str                        # "assistant" | "tool"
    content: list[dict[str, Any]]    # raw message content blocks


class Trajectory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    full_name: str
    repo_url: str
    clone_url: str
    default_branch: str
    verified_sha: str
    deselect_args: list[str]
    python_version: str

    main_task: str
    side_task: str = ""              # empty for baseline agent

    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_patch: str = ""
    error: str | None = None

    main_task_verified: bool = False
    side_task_verified: bool = False
    label: str = "baseline"          # "baseline" | "scheming"
    duration_s: float | None = None  # wall-clock seconds for the agent run

import uuid

from pydantic import BaseModel, Field


class TaskPair(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    full_name: str
    repo_url: str
    clone_url: str
    default_branch: str
    verified_sha: str
    deselect_args: list[str]

    main_task: str
    main_task_file: str
    main_task_function: str
    main_task_check_script: str
    main_task_check_valid: bool = False   # script ran in Docker without crash

    side_task: str
    side_task_type: str
    side_task_file: str
    side_task_function: str
    side_task_check_script: str
    side_task_check_hint: str
    side_task_check_valid: bool = False   # script ran in Docker, returned FAIL (task not yet done)

import uuid

from pydantic import BaseModel, Field, model_validator

from agents.deceptron.tasks.constants import classify_difficulty, task_version


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

    # Analytics / filtering
    task_version: str = ""    # "v1" or "v2" — set from side_task_type
    difficulty: str = ""      # "easy" | "medium" | "hard" — set from type + same-file flag

    @model_validator(mode="after")
    def _set_annotations(self) -> "TaskPair":
        if not self.task_version:
            self.task_version = task_version(self.side_task_type)
        if not self.difficulty:
            same_file = self.main_task_file == self.side_task_file
            self.difficulty = classify_difficulty(self.side_task_type, same_file)
        return self

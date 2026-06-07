"""Render an agent trajectory dict into monitor-readable text.

Shared by `deceptron.monitor.prompted.main` and `repro.monitor.prompted.main` — both must
format trajectories identically so that scores produced by different prompts
are comparable (the only thing that should vary is the prompt, not the input).
"""

MAX_TRAJECTORY_CHARS = 12000   # truncate very long trajectories


def format_trajectory(traj: dict) -> str:
    """Format trajectory steps into a readable string for the monitor."""
    lines = []
    lines.append(f"Repository: {traj['full_name']}")
    lines.append(f"Task: {traj['main_task']}\n")

    for step in traj.get("steps", []):
        role = step["role"]
        for block in step.get("content", []):
            btype = block.get("type", "")

            if role == "assistant":
                if btype == "text":
                    text = block["text"].strip()
                    if text:
                        lines.append(f"[Agent]: {text}")
                elif btype == "tool_use":
                    name = block["name"]
                    inp = block.get("input", {})
                    if name == "write_file":
                        lines.append(f"[Tool call]: write_file(path={inp.get('path')})")
                        content = inp.get("content", "")
                        lines.append(f"  content ({len(content)} chars):\n{content[:2000]}")
                    elif name == "read_file":
                        lines.append(f"[Tool call]: read_file(path={inp.get('path')})")
                    elif name == "list_directory":
                        lines.append(f"[Tool call]: list_directory(path={inp.get('path')})")
                    else:
                        lines.append(f"[Tool call]: {name}({inp})")

            elif role == "tool":
                content = block.get("content", "")
                if len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"[Tool result]: {content}")

    if traj.get("final_patch"):
        lines.append(f"\n[Final patch]:\n{traj['final_patch']}")

    text = "\n".join(lines)
    if len(text) > MAX_TRAJECTORY_CHARS:
        text = text[:MAX_TRAJECTORY_CHARS] + "\n... [truncated]"
    return text

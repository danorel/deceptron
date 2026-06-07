"""Strip large log fields from repos_verified.jsonl for git commit — see
`main.py` for the CLI."""

import json
from pathlib import Path

STRIP_FIELDS = {"install_log", "test_log"}
DEFAULT_SRC = "data/agents/deceptron/mining/repos_verified.jsonl"
DEFAULT_DST = "data/agents/deceptron/mining/repos_verified_clean.jsonl"


def run(src: str = DEFAULT_SRC, dst: str = DEFAULT_DST) -> None:
    src_path, dst_path = Path(src), Path(dst)
    if not src_path.exists():
        print(f"{src_path} not found. Run verifier first.")
        return

    rows = []
    with src_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                for field in STRIP_FIELDS:
                    row.pop(field, None)
                rows.append(row)

    with dst_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Written {len(rows)} repos to {dst_path} (logs stripped)")

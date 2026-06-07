"""Strip large log fields from repos_verified.jsonl for git commit.

Usage:
    python -m deceptron.scripts.strip_logs
"""

import json
from pathlib import Path

STRIP_FIELDS = {"install_log", "test_log"}
SRC = Path("data/deceptron/mining/repos_verified.jsonl")
DST = Path("data/deceptron/mining/repos_verified_clean.jsonl")


def main() -> None:
    if not SRC.exists():
        print(f"{SRC} not found. Run verifier first.")
        return

    rows = []
    with SRC.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                for field in STRIP_FIELDS:
                    row.pop(field, None)
                rows.append(row)

    with DST.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Written {len(rows)} repos to {DST} (logs stripped)")


if __name__ == "__main__":
    main()

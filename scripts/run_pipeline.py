from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("$", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    run(["scripts/generate_dataset.py", "--episodes", "120", "--size", "11"])
    run(["scripts/train_agent.py", "--epochs", "7"])
    run(["scripts/evaluate.py", "--episodes", "36", "--size", "11"])


if __name__ == "__main__":
    main()

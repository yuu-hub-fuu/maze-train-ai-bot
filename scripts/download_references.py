from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REFERENCES = {
    "maze-dataset": "https://github.com/understanding-search/maze-dataset.git",
    "neural-astar": "https://github.com/omron-sinicx/neural-astar.git",
    "pymaze": "https://github.com/jostbr/pymaze.git",
    "decision-transformer": "https://github.com/kzl/decision-transformer.git",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download optional upstream reference repositories.")
    parser.add_argument("--name", choices=list(REFERENCES) + ["all"], default="maze-dataset")
    parser.add_argument("--out-dir", default="third_party")
    args = parser.parse_args()
    names = list(REFERENCES) if args.name == "all" else [args.name]
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        dest = root / name
        if dest.exists():
            print(f"exists: {dest}")
            continue
        subprocess.run(["git", "clone", "--depth", "1", REFERENCES[name], str(dest)], check=True)


if __name__ == "__main__":
    main()

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import data


def main() -> None:
    print("raw -> interim ...", flush=True)
    data.build_interim()
    print("interim -> processed ...", flush=True)
    summary = data.prepare()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

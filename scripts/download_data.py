import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    config.RAW.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, url in config.FILES.items():
        path = config.RAW / url.rsplit("/", 1)[1]
        if path.exists():
            print(f"{name:13s} уже есть: {path.name}")
        else:
            print(f"{name:13s} скачиваю {url}", flush=True)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
        lines.append(f"{sha256(path)}  {path.name}")
        print(f"{'':13s} {path.stat().st_size / 2**20:.1f} MB")
    (config.RAW / "sha256.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

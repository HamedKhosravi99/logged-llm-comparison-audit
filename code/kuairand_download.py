"""Download KuaiRand-Pure (Zenodo 10439422, CC BY 4.0) into code/data/kuairand/.

The data is not redistributed with this repository; this script fetches it from the publisher.
Usage: python code/kuairand_download.py [target_dir]
"""
import sys, tarfile, urllib.request
from pathlib import Path

URL = "https://zenodo.org/api/records/10439422/files/KuaiRand-Pure.tar.gz/content"


def main(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    tgz = target / "KuaiRand-Pure.tar.gz"
    if not (target / "KuaiRand-Pure" / "data").exists():
        if not tgz.exists():
            print(f"downloading KuaiRand-Pure (47 MB) -> {tgz}")
            urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(target)
    out = target / "KuaiRand-Pure" / "data"
    assert (out / "log_random_4_22_to_5_08_pure.csv").exists()
    print(f"ready: {out}")
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "kuairand"
    main(root)

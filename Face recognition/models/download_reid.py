"""Fetch the official YoutuReID model and verify its published SHA-256."""
import utils.session
import concurrent.futures
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/person_reid_youtureid/person_reid_youtu_2021nov.onnx"
SIZE = 106878407
SHA256 = "0579683334d4b9440221606dcb461656dd0dc64143b18f48faedaced9b4f580d"


def fetch(part):
    start = part * SIZE // 32
    end = (part + 1) * SIZE // 32 - 1
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status != 206:
            raise RuntimeError("Model server did not support range downloads")
        data = response.read()
    if len(data) != end - start + 1:
        raise RuntimeError(f"Incomplete download part {part}")
    print(f"Downloaded part {part+1}/32", flush=True)
    return data


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        data = b"".join(pool.map(fetch, range(32)))
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("Model checksum mismatch")
    (ROOT / "person_reid_youtu_fp32.onnx").write_bytes(data)
    with urllib.request.urlopen("https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/person_reid_youtureid/LICENSE", timeout=30) as response:
        (ROOT / "YoutuReID-LICENSE.txt").write_bytes(response.read())
    print("Official body model downloaded and verified.")
